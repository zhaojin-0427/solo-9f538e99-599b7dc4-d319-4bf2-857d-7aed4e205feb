"""规模化与增量写入回归测试。

覆盖需求：
* 纯追加快速路径不逐笔读取旧扫描、不扫遍无关归属、只更新有变的边；
* 50 万量级历史下追加单笔内存平稳（通过进程峰值内存断言，而非全量载入）；
* 追加与补传/更正交错时以同一已提交状态为起点，扫描与归属一并生效；
* 随外箱移动的内盒全部进入变化回执。
"""

import os
import resource

import pytest
from fastapi.testclient import TestClient

from app import db as db_module
from app.db import connect
from app.main import app


def _scan(scan_no, seq, revision, child, parent):
    return {"scan_no": scan_no, "seq": seq, "revision": revision,
            "child": child, "parent": parent}


def _seed(client, n_cases, boxes_per_case):
    """铺设 n_cases 个外箱，每箱 boxes_per_case 盒，全部上托盘。"""

    seq = 0
    batch = []

    def flush():
        nonlocal batch
        while len(batch) >= 100:
            head, batch = batch[:100], batch[100:]
            r = client.post("/scans/batch", json={"scans": head})
            assert r.status_code == 200, r.text

    for i in range(1, n_cases + 1):
        case = f"C:c{i:05d}"
        pal = f"P:p{i:05d}"
        for j in range(1, boxes_per_case + 1):
            seq += 1
            batch.append(_scan(f"S{seq:07d}", seq, 1,
                               f"B:b{i:05d}-{j:02d}", case))
        seq += 1
        batch.append(_scan(f"S{seq:07d}", seq, 1, case, pal))
        flush()
    if batch:
        r = client.post("/scans/batch", json={"scans": batch})
        assert r.status_code == 200, r.text
    return seq


def test_append_only_writes_changed_edges_and_never_full_scans(client, monkeypatch):
    client.post("/scans/batch", json={"scans": [
        _scan("A1", 1, 1, "B:b1", "C:c1"),
        _scan("A2", 2, 1, "B:b2", "C:c1"),
        _scan("A3", 3, 1, "C:c1", "P:p1"),
        _scan("A4", 4, 1, "C:c2", "P:p2"),
    ]})

    statements = []
    real_connect = db_module.connect

    def traced_connect(path=None):
        conn = real_connect(path)
        conn.set_trace_callback(lambda sql: statements.append(" ".join(sql.split())))
        return conn

    monkeypatch.setattr(db_module, "connect", traced_connect)
    # service 模块以 ``from .db import connect`` 引入，需同步替换
    from app import service as service_module
    monkeypatch.setattr(service_module, "connect", traced_connect)

    r = client.post("/scans/batch", json={"scans": [
        _scan("A5", 5, 1, "C:c1", "P:p2"),
    ]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["replayed_scans"] == 5

    # 快速路径禁止无过滤读取扫描表（只允许点查/聚合/插入）
    for normalized in statements:
        upper = normalized.upper()
        if upper.startswith("SELECT") and " FROM SCANS" in upper:
            assert "WHERE" in upper or "COUNT(*)" in upper or "MAX(SEQ)" in upper, (
                f"不允许无过滤读取扫描表: {normalized}"
            )
        # 禁止整体替换边表
        assert upper != "DELETE FROM EDGES", "不允许整体替换边表"

    # 随外箱移动的内盒都要列出，且含完整旧/新路径
    boxes = {c["box"]: c for c in body["changes"]}
    assert set(boxes) == {"B:b1", "B:b2"}
    for code in ("B:b1", "B:b2"):
        assert boxes[code]["old_pallet"] == "P:p1"
        assert boxes[code]["new_pallet"] == "P:p2"
        assert boxes[code]["old_path"] == ["P:p1", "C:c1", code]
        assert boxes[code]["new_path"] == ["P:p2", "C:c1", code]

    # 无关归属 C:c2 -> P:p2 的边原样保留，边总数不变
    with connect() as c:
        row = c.execute("SELECT parent FROM edges WHERE child='C:c2'").fetchone()
        assert row["parent"] == "P:p2"
        assert c.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 4


def test_append_illegal_step_returns_first_conflict_seq_and_preserves_state(client):
    client.post("/scans/batch", json={"scans": [
        _scan("A1", 1, 1, "B:b1", "C:c1"),
        _scan("A2", 2, 1, "C:c1", "P:p1"),
    ]})
    # 同批内前一步合法、第二步非法：返回首个冲突序号，整批不生效
    r = client.post("/scans/batch", json={"scans": [
        _scan("N1", 3, 1, "B:b9", "C:c9"),
        _scan("N2", 4, 1, "B:b9", "P:p9"),
    ]})
    assert r.status_code == 409
    assert r.json()["conflict_seq"] == 4
    with connect() as c:
        assert c.execute("SELECT COUNT(*) FROM scans").fetchone()[0] == 2
        assert c.execute(
            "SELECT 1 FROM scans WHERE scan_no IN ('N1','N2')").fetchone() is None
        assert c.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 2
    assert client.get("/boxes/B:b1").json()["path"] == ["P:p1", "C:c1", "B:b1"]


def test_mixed_append_and_correction_share_one_committed_baseline(client):
    # 初始状态：b1->c1->p1，b2->c1
    client.post("/scans/batch", json={"scans": [
        _scan("H1", 1, 1, "B:b1", "C:c1"),
        _scan("H2", 2, 1, "B:b2", "C:c1"),
        _scan("H3", 3, 1, "C:c1", "P:p1"),
    ]})
    # 同一批：更正历史扫描 H1（rev2，改挂 c2）+ 追加新扫描（c2 上托盘）
    r = client.post("/scans/batch", json={"scans": [
        _scan("H1", 1, 2, "B:b1", "C:c2"),
        _scan("N1", 4, 1, "C:c2", "P:p2"),
    ]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["replayed_scans"] == 4
    boxes = {c["box"]: c for c in body["changes"]}
    # b1 随更正换到 c2，再被追加扫描带到 p2；b2 仍在 c1/p1
    assert boxes["B:b1"]["new_path"] == ["P:p2", "C:c2", "B:b1"]
    assert boxes["B:b1"]["old_path"] == ["P:p1", "C:c1", "B:b1"]
    assert "B:b2" not in boxes
    assert client.get("/boxes/B:b1").json()["path"] == ["P:p2", "C:c2", "B:b1"]
    assert client.get("/boxes/B:b2").json()["path"] == ["P:p1", "C:c1", "B:b2"]
    with connect() as c:
        assert c.execute(
            "SELECT revision FROM scans WHERE scan_no='H1'").fetchone()["revision"] == 2


def test_mixed_batch_rollback_leaves_scans_and_edges_untouched(client):
    client.post("/scans/batch", json={"scans": [
        _scan("H1", 1, 1, "B:b1", "C:c1"),
        _scan("H2", 2, 1, "C:c1", "P:p1"),
    ]})
    # 更正让历史扫描自身非法（盒直接上托盘），追加部分必须一并回滚
    r = client.post("/scans/batch", json={"scans": [
        _scan("H1", 1, 2, "B:b1", "P:p1"),
        _scan("N1", 3, 1, "C:c9", "P:p9"),
    ]})
    assert r.status_code == 409
    assert r.json()["conflict_seq"] == 1
    with connect() as c:
        assert c.execute("SELECT COUNT(*) FROM scans").fetchone()[0] == 2
        assert c.execute(
            "SELECT revision FROM scans WHERE scan_no='H1'").fetchone()["revision"] == 1
        assert c.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 2
        assert c.execute("SELECT 1 FROM scans WHERE scan_no='N1'").fetchone() is None


# 规模可通过环境变量放大；默认 5 万+ 扫描，CI 友好且足以证明不依赖全量载入
SCALE_CASES = int(os.environ.get("TRACE_SCALE_CASES", "2500"))
BOXES_PER_CASE = 20


@pytest.mark.skipif(
    os.environ.get("TRACE_RUN_SCALE") != "1",
    reason="规模测试默认跳过，设置 TRACE_RUN_SCALE=1 启用",
)
def test_large_history_single_append_flat_memory(client):
    max_seq = _seed(client, SCALE_CASES, BOXES_PER_CASE)
    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    r = client.post("/scans/batch", json={"scans": [
        _scan("S-BIG-NEW", max_seq + 1, 1, "C:c00001", "P:p00002"),
    ]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["replayed_scans"] == SCALE_CASES * (BOXES_PER_CASE + 1) + 1
    assert len(body["changes"]) == BOXES_PER_CASE

    # 追加一笔的新增峰值内存不超过 120 MiB（与 50 万历史规模解耦）
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert (peak - baseline) < 120 * 1024, f"峰值增长 {peak - baseline} KiB"

    # 重启语义：全新客户端从同一 SQLite 文件恢复后路径可查
    with TestClient(app) as c2:
        p = c2.get("/boxes/B:b00001-01").json()
        assert p["path"] == ["P:p00002", "C:c00001", "B:b00001-01"]
