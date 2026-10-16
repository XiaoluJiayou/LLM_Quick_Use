from typing import Optional


class PolyNode:
    def __init__(
        self,
        x: int = 0,
        y: int = 0,
        next: Optional["PolyNode"] = None
    ):
        self.coefficient = x
        self.power = y
        self.next = next


class Solution:
    def addPoly(
        self,
        poly1: Optional[PolyNode],
        poly2: Optional[PolyNode]
    ) -> Optional[PolyNode]:

        dummy = PolyNode()
        cur = dummy
        p1 = poly1
        p2 = poly2

        while p1 and p2:
            if p1.power > p2.power:
                cur.next = p1
                cur = p1
                p1 = p1.next
            elif p1.power < p2.power:
                cur.next = p2
                cur = p2
                p2 = p2.next
            else:
                x = p1.coefficient + p2.coefficient
                if x != 0:
                    node = PolyNode(x, p1.power)
                    cur.next = node
                    cur = node
                p1 = p1.next
                p2 = p2.next

        cur.next = p1 if p1 else p2
        return dummy.next


# ============================================================
# 以下都是测试辅助代码，不需要你修改
# ============================================================

def build_poly(items: list[tuple[int, int]]) -> Optional[PolyNode]:
    """
    根据 [(系数, 指数), ...] 创建多项式链表。

    例如：
    [(5, 4), (2, 2), (3, 0)]

    表示：
    5x^4 + 2x^2 + 3
    """
    dummy = PolyNode()
    cur = dummy

    for coefficient, power in items:
        cur.next = PolyNode(coefficient, power)
        cur = cur.next

    return dummy.next


def poly_to_list(head: Optional[PolyNode]) -> list[tuple[int, int]]:
    """
    把链表转换成列表，方便 assert 比较。
    """
    result = []

    cur = head
    while cur:
        result.append((cur.coefficient, cur.power))
        cur = cur.next

    return result


def print_poly(head: Optional[PolyNode]) -> None:
    """
    以比较直观的形式打印多项式。
    """
    if not head:
        print("0")
        return

    result = []
    cur = head

    while cur:
        result.append(f"({cur.coefficient})x^{cur.power}")
        cur = cur.next

    print(" + ".join(result))


def run_test(
    test_name: str,
    poly1_data: list[tuple[int, int]],
    poly2_data: list[tuple[int, int]],
    expected: list[tuple[int, int]]
) -> None:

    poly1 = build_poly(poly1_data)
    poly2 = build_poly(poly2_data)

    solution = Solution()
    result = solution.addPoly(poly1, poly2)

    actual = poly_to_list(result)

    if actual == expected:
        print(f"✅ {test_name}：通过")
    else:
        print(f"❌ {test_name}：失败")
        print(f"   期望：{expected}")
        print(f"   实际：{actual}")


if __name__ == "__main__":

    # --------------------------------------------------------
    # 测试 1：普通情况
    #
    # 5x^4 + 2x^2 + 3
    # +
    # 4x^3 - 2x^2 + 1
    #
    # =
    # 5x^4 + 4x^3 + 4
    # --------------------------------------------------------

    run_test(
        "普通合并",
        [(5, 4), (2, 2), (3, 0)],
        [(4, 3), (-2, 2), (1, 0)],
        [(5, 4), (4, 3), (4, 0)]
    )

    # --------------------------------------------------------
    # 测试 2：两个多项式指数完全错开
    # --------------------------------------------------------

    run_test(
        "指数完全不同",
        [(3, 5), (2, 3), (1, 1)],
        [(4, 4), (6, 2), (7, 0)],
        [
            (3, 5),
            (4, 4),
            (2, 3),
            (6, 2),
            (1, 1),
            (7, 0)
        ]
    )

    # --------------------------------------------------------
    # 测试 3：相同指数需要合并
    # --------------------------------------------------------

    run_test(
        "相同指数合并",
        [(3, 3), (2, 1)],
        [(5, 3), (4, 1)],
        [(8, 3), (6, 1)]
    )

    # --------------------------------------------------------
    # 测试 4：相加以后变成 0
    # --------------------------------------------------------

    run_test(
        "系数抵消",
        [(3, 3), (2, 1)],
        [(-3, 3), (5, 0)],
        [(2, 1), (5, 0)]
    )

    # --------------------------------------------------------
    # 测试 5：整个结果全部抵消
    # --------------------------------------------------------

    run_test(
        "全部抵消",
        [(3, 3), (2, 1)],
        [(-3, 3), (-2, 1)],
        []
    )

    # --------------------------------------------------------
    # 测试 6：poly1 比 poly2 长
    # --------------------------------------------------------

    run_test(
        "poly1 剩余节点",
        [(5, 5), (4, 4), (3, 3), (2, 2)],
        [(1, 5)],
        [(6, 5), (4, 4), (3, 3), (2, 2)]
    )

    # --------------------------------------------------------
    # 测试 7：poly2 比 poly1 长
    # --------------------------------------------------------

    run_test(
        "poly2 剩余节点",
        [(1, 4)],
        [(3, 5), (2, 4), (8, 2), (6, 0)],
        [(3, 5), (3, 4), (8, 2), (6, 0)]
    )

    # --------------------------------------------------------
    # 测试 8：poly1 为空
    # --------------------------------------------------------

    run_test(
        "poly1 为空",
        [],
        [(4, 3), (2, 1)],
        [(4, 3), (2, 1)]
    )

    # --------------------------------------------------------
    # 测试 9：poly2 为空
    # --------------------------------------------------------

    run_test(
        "poly2 为空",
        [(4, 3), (2, 1)],
        [],
        [(4, 3), (2, 1)]
    )

    # --------------------------------------------------------
    # 测试 10：两个都为空
    # --------------------------------------------------------

    run_test(
        "两个都为空",
        [],
        [],
        []
    )






