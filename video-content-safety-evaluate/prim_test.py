class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        graph = [[float("inf")] * n for _ in range(n)]

        for x, y, cost in times:
            graph[x - 1][y - 1] = cost

        min_dist = [float("inf")] * n
        in_dist = [False] * n

        min_dist[k - 1] = 0

        for _ in range(n):
            u = -1

            for i in range(n):
                if not in_dist[i] and (u == -1 or min_dist[i] < min_dist[u]):
                    u = i

            if u == -1:
                break

            in_dist[u] = True

            for j in range(n):
                if not in_dist[j] and graph[u][j] < float("inf") and graph[u][j] + min_dist[u] < min_dist[j]:
                    min_dist[j] = graph[u][j] + min_dist[u]

        res = max(min_dist)

        return res if res < float("inf") else -1

