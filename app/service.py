"""核心业务逻辑：扫描修订合并、按序号重放、层级校验与路径计算。

设计要点
--------
1. 每个扫描号只保留最高修订；同号同修订异文整批拒绝，低修订整批拒绝，
   同号同修订同文重提为幂等空操作。
2. 两条提交路径，都在单个 SQLite 事务内完成，失败即回滚：

   * **纯追加快速路径**：批内扫描号均未用过、且序号均大于库中最大生效序号。
     只做索引点查（按 ``child`` 查当前父项、按 ``parent`` 展开子树），不逐笔
     读取旧扫描、不扫遍无关归属；边表只 upsert/delete 真正变化的行。50 万笔
     历史下追加一笔的内存占用与库规模无关。
   * **通用重放路径**（补传/更正/重传，或与追加交错）：把待合并的扫描写入
     连接级临时表，由 SQLite 按 ``seq`` 全序流式输出重放，Python 侧每次只保留
     当前一行；新层级落在临时边表，最后只把**有差异**的边同步到正式表，
     不做全表替换。排序等大内存操作通过 ``temp_store=FILE`` 落盘。
3. 重放中发现的首个非法序号通过 ``ReplayConflict.conflict_seq`` 返回，
   原扫描与归属保持不变。
"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .db import connect, transaction, write_lock
from .schemas import BatchIn, ScanIn


class BatchError(Exception):
    """批次自身的静态校验错误（HTTP 400）。"""

    def __init__(self, code: str, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.extra = extra


class ReplayConflict(Exception):
    """重放层级规则时的冲突（HTTP 409），带首个冲突序号。"""

    def __init__(self, conflict_seq: int, reason: str) -> None:
        super().__init__(reason)
        self.conflict_seq = conflict_seq
        self.reason = reason


class RevisionConflict(Exception):
    """扫描号修订冲突（HTTP 409）。"""

    def __init__(self, scan_no: str, seq: Optional[int], reason: str) -> None:
        super().__init__(reason)
        self.scan_no = scan_no
        self.seq = seq
        self.reason = reason


def kind(code: str) -> str:
    """返回包装层级角色：'B' 盒 / 'C' 箱 / 'P' 托盘。"""

    return code[0]


# --------------------------------------------------------------------------- #
# 层级视图：把“已提交边表”和“本批覆盖后的虚拟边表”统一成点查接口
# --------------------------------------------------------------------------- #
class EdgeView:
    """只读边视图，所有访问都是主键/索引点查，不持有全表数据。"""

    def __init__(self, conn: sqlite3.Connection, table: str) -> None:
        if table not in ("edges", "temp_edge"):
            raise ValueError(f"非法边表名: {table}")
        self._conn = conn
        self._table = table
        self._parent_cache: Dict[str, Optional[str]] = {}
        self._children_cache: Dict[str, Tuple[str, ...]] = {}

    def parent(self, code: str) -> Optional[str]:
        if code not in self._parent_cache:
            row = self._conn.execute(
                f"SELECT parent FROM {self._table} WHERE child = ?", (code,)
            ).fetchone()
            self._parent_cache[code] = row[0] if row is not None else None
        return self._parent_cache[code]

    def children(self, code: str) -> Tuple[str, ...]:
        if code not in self._children_cache:
            rows = self._conn.execute(
                f"SELECT child FROM {self._table} WHERE parent = ? ORDER BY child",
                (code,),
            ).fetchall()
            self._children_cache[code] = tuple(r[0] for r in rows)
        return self._children_cache[code]

    def path_of(self, code: str) -> List[str]:
        """自顶（托盘/最外层）向底（该码）返回完整包装路径。"""

        chain = [code]
        cur = self.parent(code)
        seen = {code}
        while cur is not None:
            if cur in seen:  # 防御性：正常流程重放时已保证无环
                raise ReplayConflict(-1, "包装层级存在环")
            seen.add(cur)
            chain.append(cur)
            cur = self.parent(cur)
        chain.reverse()
        return chain

    def pallet_of(self, code: str) -> Optional[str]:
        """返回当前所在托盘（路径上首个 P: 节点），未上托盘返回 None。"""

        for node in self.path_of(code):
            if node.startswith("P:"):
                return node
        return None

    def descendant_boxes(self, root: str) -> Iterator[str]:
        """自顶向下遍历 root 子树中的全部 B: 盒（含 root 自身若是盒）。"""

        stack: List[str] = [root]
        seen = set()
        while stack:
            node = stack.pop()
            if node in seen:  # 防御性环保护
                continue
            seen.add(node)
            if node.startswith("B:"):
                yield node
            stack.extend(self.children(node))


class OverlayView:
    """在已提交边表之上叠加本批变更（child -> 新父项 / None=拆出）的虚拟视图。

    构造期间不读取整张边表：``parent``/``children`` 全部退化为点查；
    仅缓存本批实际触及的少量节点。
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        overrides: Dict[str, Optional[str]],
        memoize: bool = True,
    ) -> None:
        self._conn = conn
        self._overrides = overrides
        self._memo = memoize
        self._parent_cache: Dict[str, Optional[str]] = {}
        self._children_cache: Dict[str, Tuple[str, ...]] = {}

    def parent(self, code: str) -> Optional[str]:
        if code in self._overrides:
            return self._overrides[code]
        if self._memo and code in self._parent_cache:
            return self._parent_cache[code]
        row = self._conn.execute(
            "SELECT parent FROM edges WHERE child = ?", (code,)
        ).fetchone()
        value = row[0] if row is not None else None
        if self._memo:
            self._parent_cache[code] = value
        return value

    def children(self, code: str) -> Tuple[str, ...]:
        if self._memo and code in self._children_cache:
            return self._children_cache[code]
        # 已提交子项
        rows = self._conn.execute(
            "SELECT child FROM edges WHERE parent = ? ORDER BY child", (code,)
        ).fetchall()
        kids = {r[0] for r in rows}
        # 叠加本批：新挂入的加入、改挂/拆出的移除
        for child, parent in self._overrides.items():
            if parent == code:
                kids.add(child)
            elif child in kids:
                kids.discard(child)
        result = tuple(sorted(kids))
        if self._memo:
            self._children_cache[code] = result
        return result

    def path_of(self, code: str) -> List[str]:
        chain = [code]
        cur = self.parent(code)
        seen = {code}
        while cur is not None:
            if cur in seen:
                raise ReplayConflict(-1, "包装层级存在环")
            seen.add(cur)
            chain.append(cur)
            cur = self.parent(cur)
        chain.reverse()
        return chain

    def pallet_of(self, code: str) -> Optional[str]:
        for node in self.path_of(code):
            if node.startswith("P:"):
                return node
        return None

    def descendant_boxes(self, root: str) -> Iterator[str]:
        stack: List[str] = [root]
        seen = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            if node.startswith("B:"):
                yield node
            stack.extend(self.children(node))


def path_of(edges: Dict[str, str], code: str) -> List[str]:
    """字典边表上的路径计算（保留给纯函数场景/测试使用）。"""

    chain = [code]
    cur = edges.get(code)
    seen = {code}
    while cur is not None:
        if cur in seen:
            raise ReplayConflict(-1, "包装层级存在环")
        seen.add(cur)
        chain.append(cur)
        cur = edges.get(cur)
    chain.reverse()
    return chain


def pallet_of(edges: Dict[str, str], code: str) -> Optional[str]:
    """字典边表上的托盘查询（保留给纯函数场景/测试使用）。"""

    for node in path_of(edges, code):
        if node.startswith("P:"):
            return node
    return None


# --------------------------------------------------------------------------- #
# 批次静态校验与角色/环规则
# --------------------------------------------------------------------------- #
def validate_batch(batch: BatchIn) -> None:
    """批次内静态校验：非空、扫描号不重复、序号不重复。"""

    if not batch.scans:
        raise BatchError("empty_batch", "每批至少提交 1 笔扫描")

    scan_nos: Dict[str, ScanIn] = {}
    seqs: Dict[int, ScanIn] = {}
    for item in batch.scans:
        if item.scan_no in scan_nos:
            raise BatchError(
                "duplicate_scan_no",
                f"批次内扫描号重复: {item.scan_no}",
                scan_no=item.scan_no,
            )
        scan_nos[item.scan_no] = item

        if item.seq in seqs:
            raise BatchError(
                "duplicate_seq",
                f"批次内序号重复: {item.seq}",
                conflict_seq=item.seq,
            )
        seqs[item.seq] = item


def _check_role(seq: int, child: str, parent: str) -> None:
    """包装角色规则：盒只能入箱；箱可入箱/托盘；托盘不作子项。"""

    child_kind = kind(child)
    parent_kind = kind(parent)

    if child_kind == "P":
        raise ReplayConflict(seq, f"托盘 {child} 不能作为子项装入 {parent}")
    if child_kind == "B" and parent_kind != "C":
        raise ReplayConflict(seq, f"盒 {child} 只能装入 C: 开头的箱，不能装入 {parent}")
    if child_kind == "C" and parent_kind not in ("C", "P"):
        raise ReplayConflict(seq, f"箱 {child} 只能装入另一箱或托盘，不能装入 {parent}")


def _check_cycle(seq: int, parent_of, child: str, parent: str) -> None:
    """若把 child 挂到 parent 下会形成环，则按当前序号拒绝。

    ``parent_of`` 为可调用对象，封装当前（可能含本批覆盖的）层级。
    """

    cur: Optional[str] = parent
    seen = set()
    while cur is not None:
        if cur == child:
            raise ReplayConflict(seq, f"装入 {parent} 会使 {child} 的层级形成环")
        if cur in seen:  # 既有层级理论上无环，防御性截断
            break
        seen.add(cur)
        cur = parent_of(cur)


# --------------------------------------------------------------------------- #
# 回执：只列路径变化的盒码（随外箱移动的内盒自然包含在内）
# --------------------------------------------------------------------------- #
def _collect_changes(
    conn: sqlite3.Connection,
    touched_children: List[str],
    new_view: Any,
) -> List[Dict[str, Any]]:
    """根据变更边的子项，收集旧/新路径不同的盒码。

    旧状态查正式 ``edges`` 表，新状态由调用方提供（OverlayView 或临时表视图）；
    每个变更子项只展开其旧子树与新子树，无关归属不读取。
    """

    old_view = EdgeView(conn, "edges")

    candidates: set[str] = set()
    for child in touched_children:
        candidates.update(old_view.descendant_boxes(child))
        candidates.update(new_view.descendant_boxes(child))

    changes: List[Dict[str, Any]] = []
    for box in sorted(candidates):
        old_attached = old_view.parent(box) is not None
        new_attached = new_view.parent(box) is not None
        old_path = old_view.path_of(box) if old_attached else [box]
        new_path = new_view.path_of(box) if new_attached else [box]
        if old_path == new_path:
            continue
        changes.append(
            {
                "box": box,
                "old_pallet": old_view.pallet_of(box) if old_attached else None,
                "new_pallet": new_view.pallet_of(box) if new_attached else None,
                "old_path": old_path,
                "new_path": new_path,
            }
        )
    return changes


# --------------------------------------------------------------------------- #
# 纯追加快速路径
# --------------------------------------------------------------------------- #
def _try_append_fast_path(
    conn: sqlite3.Connection, scans: List[ScanIn]
) -> Optional[Dict[str, Any]]:
    """满足纯追加条件时走快速路径，否则返回 None 交由通用重放处理。"""

    max_row = conn.execute("SELECT MAX(seq) FROM scans").fetchone()
    max_seq: Optional[int] = max_row[0] if max_row is not None else None

    # 条件 1：批内扫描号均未用过（一次点查，不逐笔拉取旧扫描内容）
    placeholders = ",".join("?" for _ in scans)
    existing_no = conn.execute(
        f"SELECT scan_no FROM scans WHERE scan_no IN ({placeholders})",
        [s.scan_no for s in scans],
    ).fetchall()
    if existing_no:
        return None

    # 条件 2：批内序号均大于库中最大生效序号
    if max_seq is not None and any(s.seq <= max_seq for s in scans):
        return None

    # 按序号逐步校验，首个非法操作以其序号作为 conflict_seq
    ordered = sorted(scans, key=lambda s: s.seq)
    # 校验期视图不缓存父向链，确保环检查始终看到本批此前各步的最新覆盖；
    # 视图持有同一个 overrides 字典的引用，逐步写入即可被后续步骤观察到。
    overrides: Dict[str, Optional[str]] = {}
    validate_view = OverlayView(conn, overrides, memoize=False)
    for item in ordered:
        if item.parent is None:
            # 拆出：无归属时也是合法空操作
            overrides[item.child] = None
            continue
        _check_role(item.seq, item.child, item.parent)
        _check_cycle(item.seq, validate_view.parent, item.child, item.parent)
        overrides[item.child] = item.parent

    # 只保留与已提交状态确实不同的归属
    real_changes: List[Tuple[str, Optional[str]]] = []
    for child, new_parent in overrides.items():
        row = conn.execute(
            "SELECT parent FROM edges WHERE child = ?", (child,)
        ).fetchone()
        old_parent = row[0] if row is not None else None
        if old_parent != new_parent:
            real_changes.append((child, new_parent))

    # 回执必须在写入边表之前计算：OverlayView 给出新状态，正式表仍是旧状态
    receipt_view = OverlayView(conn, overrides, memoize=True)
    changes = _collect_changes(conn, [c for c, _ in real_changes], receipt_view)

    # 写入：仅更新有变的归属，不触碰无关边
    for child, new_parent in real_changes:
        if new_parent is None:
            conn.execute("DELETE FROM edges WHERE child = ?", (child,))
        else:
            conn.execute(
                """
                INSERT INTO edges(child, parent) VALUES (?, ?)
                ON CONFLICT(child) DO UPDATE SET parent = excluded.parent
                """,
                (child, new_parent),
            )

    conn.executemany(
        """
        INSERT INTO scans(scan_no, seq, revision, child, parent)
        VALUES (:scan_no, :seq, :revision, :child, :parent)
        ON CONFLICT(scan_no) DO UPDATE SET
            seq = excluded.seq,
            revision = excluded.revision,
            child = excluded.child,
            parent = excluded.parent,
            updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        """,
        [item.model_dump() for item in ordered],
    )

    total = conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
    return {"accepted": True, "replayed_scans": total, "changes": changes}


# --------------------------------------------------------------------------- #
# 通用重放路径（补传 / 更正 / 重传，或与追加交错）
# --------------------------------------------------------------------------- #
_TEMP_DDL = """
CREATE TEMP TABLE IF NOT EXISTS temp_scan (
    scan_no  TEXT PRIMARY KEY,
    seq      INTEGER NOT NULL,
    revision INTEGER NOT NULL,
    child    TEXT NOT NULL,
    parent   TEXT
);
CREATE TEMP TABLE IF NOT EXISTS temp_edge (
    child  TEXT PRIMARY KEY,
    parent TEXT NOT NULL
);
"""


def _replay_general(
    conn: sqlite3.Connection, scans: List[ScanIn]
) -> Dict[str, Any]:
    # 连接级临时表可能是上一次事务的残留，先清空
    conn.execute("DELETE FROM temp_scan")
    conn.execute("DELETE FROM temp_edge")

    # 修订规则逐笔校验（只点查涉及的扫描号）
    incoming = {item.scan_no: item for item in scans}
    placeholders = ",".join("?" for _ in incoming)
    old_rows = conn.execute(
        f"SELECT scan_no, seq, revision, child, parent FROM scans "
        f"WHERE scan_no IN ({placeholders})",
        list(incoming),
    ).fetchall()

    idempotent = True
    for row in old_rows:
        item = incoming[row["scan_no"]]
        if item.revision < row["revision"]:
            raise RevisionConflict(
                item.scan_no,
                item.seq,
                f"修订号必须递增：已存修订 {row['revision']}，收到 {item.revision}",
            )
        if item.revision == row["revision"] and (
            item.child != row["child"]
            or item.parent != row["parent"]
            or item.seq != row["seq"]
        ):
            raise RevisionConflict(
                item.scan_no,
                item.seq,
                f"扫描号 {item.scan_no} 同修订 {item.revision} 内容不一致，整批拒绝",
            )
        if item.revision != row["revision"]:
            idempotent = False
    if len(old_rows) < len(incoming):
        idempotent = False

    # 纯重传（全部同号同修订同文）：幂等空操作，直接以当前已提交状态出回执
    if idempotent:
        conn.executemany(
            """
            INSERT INTO scans(scan_no, seq, revision, child, parent)
            VALUES (:scan_no, :seq, :revision, :child, :parent)
            ON CONFLICT(scan_no) DO UPDATE SET
                seq = excluded.seq,
                revision = excluded.revision,
                child = excluded.child,
                parent = excluded.parent,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            """,
            [item.model_dump() for item in scans],
        )
        total = conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
        return {"accepted": True, "replayed_scans": total, "changes": []}

    conn.executemany(
        "INSERT INTO temp_scan(scan_no, seq, revision, child, parent) "
        "VALUES (:scan_no, :seq, :revision, :child, :parent)",
        [item.model_dump() for item in scans],
    )

    # 合并视图：本批扫描覆盖同号旧版，其余已存扫描原样参与；
    # 排序交给 SQLite（大结果落磁盘临时区），Python 流式逐行消费。
    merged_stream = conn.execute(
        """
        SELECT scan_no, seq, revision, child, parent FROM temp_scan
        UNION ALL
        SELECT s.scan_no, s.seq, s.revision, s.child, s.parent
        FROM scans AS s
        WHERE s.scan_no NOT IN (SELECT scan_no FROM temp_scan)
        ORDER BY seq, scan_no
        """
    )

    prev_seq: Optional[int] = None
    prev_owner: Optional[str] = None

    def _temp_parent(code: str) -> Optional[str]:
        r = conn.execute(
            "SELECT parent FROM temp_edge WHERE child = ?", (code,)
        ).fetchone()
        return r[0] if r is not None else None

    while True:
        rec = merged_stream.fetchone()
        if rec is None:
            break
        seq, scan_no, child, parent = rec["seq"], rec["scan_no"], rec["child"], rec["parent"]

        # 序号全局唯一（合并后同 seq 必属不同扫描号）
        if prev_seq is not None and seq == prev_seq and scan_no != prev_owner:
            raise ReplayConflict(
                seq, f"序号 {seq} 已被扫描号 {prev_owner} 占用"
            )
        prev_seq, prev_owner = seq, scan_no

        if parent is None:
            conn.execute("DELETE FROM temp_edge WHERE child = ?", (child,))
            continue

        _check_role(seq, child, parent)
        _check_cycle(seq, _temp_parent, child, parent)
        conn.execute(
            """
            INSERT INTO temp_edge(child, parent) VALUES (?, ?)
            ON CONFLICT(child) DO UPDATE SET parent = excluded.parent
            """,
            (child, parent),
        )

    # 只把有差异的边同步到正式表（新增/改挂/拆出），不做全表替换
    diff_upserts = conn.execute(
        """
        SELECT t.child, t.parent FROM temp_edge AS t
        LEFT JOIN edges AS e ON e.child = t.child
        WHERE e.child IS NULL OR e.parent != t.parent
        """
    ).fetchall()
    diff_deletes = conn.execute(
        """
        SELECT e.child FROM edges AS e
        LEFT JOIN temp_edge AS t ON t.child = e.child
        WHERE t.child IS NULL
        """
    ).fetchall()

    # 回执在同步正式边表之前计算：temp_edge 为新状态，edges 仍为旧状态
    touched = [r[0] for r in diff_upserts] + [r[0] for r in diff_deletes]
    new_view = EdgeView(conn, "temp_edge")
    changes = _collect_changes(conn, touched, new_view)

    conn.executemany(
        """
        INSERT INTO edges(child, parent) VALUES (?, ?)
        ON CONFLICT(child) DO UPDATE SET parent = excluded.parent
        """,
        [(r[0], r[1]) for r in diff_upserts],
    )
    conn.executemany(
        "DELETE FROM edges WHERE child = ?",
        [(r[0],) for r in diff_deletes],
    )

    # 扫描表 upsert（同号最高修订覆盖；新扫描号插入）
    conn.executemany(
        """
        INSERT INTO scans(scan_no, seq, revision, child, parent)
        VALUES (:scan_no, :seq, :revision, :child, :parent)
        ON CONFLICT(scan_no) DO UPDATE SET
            seq = excluded.seq,
            revision = excluded.revision,
            child = excluded.child,
            parent = excluded.parent,
            updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        """,
        [item.model_dump() for item in scans],
    )

    total = conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
    return {"accepted": True, "replayed_scans": total, "changes": changes}


# --------------------------------------------------------------------------- #
# 对外入口
# --------------------------------------------------------------------------- #
def submit_batch(batch: BatchIn) -> Dict[str, Any]:
    """提交一批扫描；成功返回盒码路径变化，失败保证旧状态不变。"""

    validate_batch(batch)

    with write_lock, transaction() as conn:
        # 临时表需在事务连接内确保存在（连接关闭即自动销毁）
        conn.executescript(_TEMP_DDL)

        result = _try_append_fast_path(conn, batch.scans)
        if result is not None:
            return result
        return _replay_general(conn, batch.scans)


def get_box_path(box: str) -> Optional[Dict[str, Any]]:
    """按盒查询完整当前包装路径；盒从未出现或已彻底移除返回 None。"""

    if not box.startswith("B:"):
        raise BatchError("bad_box_code", f"{box} 不是 B: 开头的盒码")

    conn = connect()
    try:
        # 索引点查：盒是否曾出现，以及当前归属链，不读取无关边
        row = conn.execute(
            "SELECT 1 FROM scans WHERE child = ? LIMIT 1", (box,)
        ).fetchone()
        if row is None:
            return None
        view = EdgeView(conn, "edges")
        if view.parent(box) is None:  # 最后一次生效操作是拆出
            return None
        path = view.path_of(box)
        return {
            "box": box,
            "parent": view.parent(box),
            "pallet": view.pallet_of(box),
            "path": path,
        }
    finally:
        conn.close()
