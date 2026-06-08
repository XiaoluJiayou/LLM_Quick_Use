"""
判断两个单链表是否相交，并返回相交的起始节点。

核心思路：
    如果两个链表相交，那么从相交点开始到末尾的所有节点都是共享的（Y 字形）。
    这意味着两个链表的尾节点一定相同。

方法一：双指针法（O(m+n) 时间，O(1) 空间）
    让两个指针分别从两个链表头部出发，每次各走一步。
    当某个指针走到末尾时，让它从另一个链表的头部重新开始。
    如果相交，两个指针最终会在相交点相遇；
    如果不相交，两个指针会同时走到 None。

方法二：HashSet 法（O(m+n) 时间，O(m) 空间）
    遍历链表 A，将所有节点存入集合。
    再遍历链表 B，第一个在集合中出现的节点就是相交点。
"""


class ListNode:
    """单链表节点定义"""

    def __init__(self, value):
        self.value = value
        self.next = None


# ============================================================================
# 方法一：双指针法（推荐）
# ============================================================================

def get_intersection_node_two_pointers(head_a, head_b):
    """
    使用双指针法找到两个链表的相交起始节点。

    算法流程：
        1. 指针 pa 从 head_a 出发，指针 pb 从 head_b 出发
        2. 两者同时向后移动，每次一步
        3. 当 pa 走到链表 A 的末尾时，将它重新指向 head_b
        4. 当 pb 走到链表 B 的末尾时，将它重新指向 head_a
        5. 如果两个链表相交，pa 和 pb 会在相交点相遇
        6. 如果不相交，两者最终会同时为 None

    直观解释：
        链表 A: a1 → a2 → c1 → c2 → c3
        链表 B: b1 → b2 → b3 → c1 → c2 → c3

        pa 走过的路径: a1 → a2 → c1 → c2 → c3 → b1 → b2 → b3 → c1
        pb 走过的路径: b1 → b2 → b3 → c1 → c2 → c3 → a1 → a2 → c1

        可以看出，当 pa 和 pb 各自走完"自己 + 对方"的路径后，
        它们走过的节点数完全相同，因此会在相交点 c1 处相遇。
    """
    if head_a is None or head_b is None:
        return None

    pa = head_a
    pb = head_b

    # 当 pa 和 pb 相遇（指向同一个节点）或者同时为 None 时退出循环
    while pa is not pb:
        # pa 走到末尾后，切换到链表 B 的头部
        if pa is None:
            pa = head_b
        else:
            pa = pa.next

        # pb 走到末尾后，切换到链表 A 的头部
        if pb is None:
            pb = head_a
        else:
            pb = pb.next

    # 此时 pa == pb，要么是相交节点，要么是 None（不相交）
    return pa


# ============================================================================
# 方法二：HashSet 法（更直观，但需要额外空间）
# ============================================================================

def get_intersection_node_hashset(head_a, head_b):
    """
    使用哈希集合找到两个链表的相交起始节点。

    算法流程：
        1. 遍历链表 A，将所有节点加入集合 visited
        2. 遍历链表 B，检查每个节点是否已在 visited 中
        3. 第一个命中的节点就是相交点
        4. 如果遍历完 B 都没有命中，说明不相交
    """
    if not head_a or not head_b:
        return None

    # 第一步：将链表 A 的所有节点加入集合
    visited = set()
    cur = head_a
    while cur:
        visited.add(cur)
        cur = cur.next

    # 第二步：遍历链表 B，查找第一个已访问过的节点
    cur = head_b
    while cur:
        if cur in visited:
            return cur
        cur = cur.next

    # 没有找到相交节点
    return None

def get_intersection_node_length_based(head_a, head_b):
    """
    方法三：长度对齐法
    主要思路：
      1. 遍历两个链表，分别统计长度及最后一个节点
      2. 如果最后节点不同，则两个链表一定不相交，返回 None
      3. 将较长链表的指针先向前移动长度之差
      4. 然后两个指针同时移动，遇到第一个相同节点即为相交点
    """
    if not head_a or not head_b:
        return None

    # 1. 统计两个链表的长度和最后节点
    cur_a, len_a = head_a, 0
    while cur_a:
        len_a += 1
        if cur_a.next is None:
            tail_a = cur_a
        cur_a = cur_a.next

    cur_b, len_b = head_b, 0
    while cur_b:
        len_b += 1
        if cur_b.next is None:
            tail_b = cur_b
        cur_b = cur_b.next

    # 2. 如果最后一个节点不是同一个，说明不相交
    if tail_a is not tail_b:
        return None

    # 3. 让两个链表的指针从“等距离末尾”的地方齐头并进
    cur_a = head_a
    cur_b = head_b
    if len_a > len_b:
        for _ in range(len_a - len_b):
            cur_a = cur_a.next
    else:
        for _ in range(len_b - len_a):
            cur_b = cur_b.next

    # 4. 同步前进，遇到第一个相同节点即为交点
    while cur_a is not cur_b:
        cur_a = cur_a.next
        cur_b = cur_b.next

    return cur_a  # 或 cur_b，都是相交节点或 None


# ============================================================================
# 辅助函数
# ============================================================================

def build_linked_list(values):
    """根据值列表构建链表，返回头节点"""
    if not values:
        return None

    head = ListNode(values[0])
    current = head
    for value in values[1:]:
        current.next = ListNode(value)
        current = current.next
    return head


def list_to_values(head):
    """将链表转换为值列表，用于打印和断言"""
    result = []
    current = head
    while current is not None:
        result.append(current.value)
        current = current.next
    return result


def create_intersecting_lists(list_a_values, list_b_values, intersect_values):
    """
    创建两个相交的链表。

    参数:
        list_a_values:  链表 A 独有的节点值列表
        list_b_values:  链表 B 独有的节点值列表
        intersect_values: 相交部分的节点值列表

    返回:
        (head_a, head_b, intersect_head): 链表 A 的头、链表 B 的头、相交起始节点
    """
    # 构建相交部分（共享的节点）
    intersect_head = build_linked_list(intersect_values)

    # 构建链表 A
    head_a = build_linked_list(list_a_values)
    if head_a is not None:
        # 找到链表 A 的尾节点
        tail_a = head_a
        while tail_a.next is not None:
            tail_a = tail_a.next
        # 将尾节点连接到相交部分
        tail_a.next = intersect_head
    else:
        head_a = intersect_head

    # 构建链表 B
    head_b = build_linked_list(list_b_values)
    if head_b is not None:
        # 找到链表 B 的尾节点
        tail_b = head_b
        while tail_b.next is not None:
            tail_b = tail_b.next
        # 将尾节点连接到相交部分
        tail_b.next = intersect_head
    else:
        head_b = intersect_head

    return head_a, head_b, intersect_head


# ============================================================================
# 测试用例
# ============================================================================

def run_tests():
    """运行所有测试用例，验证两种方法的正确性"""

    test_count = 0
    passed_count = 0

    def assert_equal(actual, expected, test_name):
        """断言工具函数"""
        nonlocal test_count, passed_count
        test_count = test_count + 1
        if actual is expected:
            passed_count = passed_count + 1
            print(f"  ✓ {test_name}")
        else:
            actual_val = actual.value if actual is not None else "None"
            expected_val = expected.value if expected is not None else "None"
            print(f"  ✗ {test_name} — 实际: {actual_val}, 期望: {expected_val}")

    print("=" * 60)
    print("测试：链表相交判断")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 测试 1：正常相交（Y 字形）
    # ------------------------------------------------------------------
    print("\n【测试组 1】正常相交")
    head_a, head_b, intersect_head = create_intersecting_lists(
        list_a_values=[1, 2],
        list_b_values=[4, 5, 6],
        intersect_values=[8, 9, 10]
    )
    print(f"  链表 A: {list_to_values(head_a)}")
    print(f"  链表 B: {list_to_values(head_b)}")
    print(f"  相交节点值: {intersect_head.value}")

    result1 = get_intersection_node_two_pointers(head_a, head_b)
    result2 = get_intersection_node_hashset(head_a, head_b)
    assert_equal(result1, intersect_head, "双指针法 - 正常相交")
    assert_equal(result2, intersect_head, "HashSet 法 - 正常相交")

    # ------------------------------------------------------------------
    # 测试 2：不相交
    # ------------------------------------------------------------------
    print("\n【测试组 2】不相交")
    head_a = build_linked_list([1, 2, 3])
    head_b = build_linked_list([4, 5, 6])
    print(f"  链表 A: {list_to_values(head_a)}")
    print(f"  链表 B: {list_to_values(head_b)}")

    result1 = get_intersection_node_two_pointers(head_a, head_b)
    result2 = get_intersection_node_hashset(head_a, head_b)
    assert_equal(result1, None, "双指针法 - 不相交")
    assert_equal(result2, None, "HashSet 法 - 不相交")

    # ------------------------------------------------------------------
    # 测试 3：一个链表为空
    # ------------------------------------------------------------------
    print("\n【测试组 3】一个链表为空")
    head_a = build_linked_list([1, 2, 3])
    head_b = None
    print(f"  链表 A: {list_to_values(head_a)}")
    print(f"  链表 B: {list_to_values(head_b)}")

    result1 = get_intersection_node_two_pointers(head_a, head_b)
    result2 = get_intersection_node_hashset(head_a, head_b)
    assert_equal(result1, None, "双指针法 - 链表 B 为空")
    assert_equal(result2, None, "HashSet 法 - 链表 B 为空")

    # ------------------------------------------------------------------
    # 测试 4：从头节点开始相交（其中一个链表完全包含在另一个中）
    # ------------------------------------------------------------------
    print("\n【测试组 4】从头节点开始相交")
    head_a, head_b, intersect_head = create_intersecting_lists(
        list_a_values=[],
        list_b_values=[4, 5],
        intersect_values=[6, 7, 8]
    )
    print(f"  链表 A: {list_to_values(head_a)}")
    print(f"  链表 B: {list_to_values(head_b)}")
    print(f"  相交节点值: {intersect_head.value}")

    result1 = get_intersection_node_two_pointers(head_a, head_b)
    result2 = get_intersection_node_hashset(head_a, head_b)
    assert_equal(result1, intersect_head, "双指针法 - 从头相交")
    assert_equal(result2, intersect_head, "HashSet 法 - 从头相交")

    # ------------------------------------------------------------------
    # 测试 5：两个链表完全相同
    # ------------------------------------------------------------------
    print("\n【测试组 5】两个链表完全相同")
    head_a = build_linked_list([1, 2, 3, 4])
    head_b = head_a
    print(f"  链表 A: {list_to_values(head_a)}")
    print(f"  链表 B: {list_to_values(head_b)}")
    print(f"  相交节点值: {head_a.value}")

    result1 = get_intersection_node_two_pointers(head_a, head_b)
    result2 = get_intersection_node_hashset(head_a, head_b)
    assert_equal(result1, head_a, "双指针法 - 两链表相同")
    assert_equal(result2, head_a, "HashSet 法 - 两链表相同")

    # ------------------------------------------------------------------
    # 测试 6：相交在最后一个节点
    # ------------------------------------------------------------------
    print("\n【测试组 6】相交在最后一个节点")
    head_a, head_b, intersect_head = create_intersecting_lists(
        list_a_values=[1, 2, 3],
        list_b_values=[4, 5, 6, 7],
        intersect_values=[99]
    )
    print(f"  链表 A: {list_to_values(head_a)}")
    print(f"  链表 B: {list_to_values(head_b)}")
    print(f"  相交节点值: {intersect_head.value}")

    result1 = get_intersection_node_two_pointers(head_a, head_b)
    result2 = get_intersection_node_hashset(head_a, head_b)
    assert_equal(result1, intersect_head, "双指针法 - 尾节点相交")
    assert_equal(result2, intersect_head, "HashSet 法 - 尾节点相交")

    # ------------------------------------------------------------------
    # 测试 7：两个链表都为空
    # ------------------------------------------------------------------
    print("\n【测试组 7】两个链表都为空")
    result1 = get_intersection_node_two_pointers(None, None)
    result2 = get_intersection_node_hashset(None, None)
    assert_equal(result1, None, "双指针法 - 两链表都为空")
    assert_equal(result2, None, "HashSet 法 - 两链表都为空")

    # ------------------------------------------------------------------
    # 测试总结
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"测试完成: {passed_count}/{test_count} 通过")
    if passed_count == test_count:
        print("所有测试用例均通过！")
    else:
        print(f"有 {test_count - passed_count} 个测试未通过，请检查代码。")
    print(f"{'=' * 60}")





if __name__ == "__main__":
    run_tests()
