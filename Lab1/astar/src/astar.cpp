#include <format>
#include <fstream>
#include <iostream>
#include <queue>
#include <tuple>
#include <unordered_set>
#include <vector>

using std::priority_queue;
using std::unordered_set;
using std::vector;

int n; // size of the map

class solve {
	class state {
	public:
		int g = 0;
		const state* parent = nullptr;
		vector<vector<bool>> data;

		int f() const { return g + h(); }

		// count the number of 1
		int loss() const {
			int res = 0;
			for (const auto& line : data)
				for (const auto& point : line)
					res += point;
			return res / 3;
		}

		int h() const {
			return loss() / 3;
		}

		bool operator<(const state& ano) const {
			return f() < ano.f();
		}

		bool operator==(const state& ano) const {
			return data == ano.data;
		}

		// turn with method s at point (x, y)
		void toward(int x, int y, int s) {
			constexpr int x_trans[][3] = {
				{0, 0, -1},
				{0, -1, 0},
				{0, 0, 1},
				{0, 1, 0}
			};
			constexpr int y_trans[][3] = {
				{0, 1, 0},
				{0, 0, -1},
				{0, -1, 0},
				{0, 0, 1}
			};
			for (int i = 0; i < 3; ++i) {
				int new_x = x + x_trans[s][i];
				int new_y = y + y_trans[s][i];
				if (new_x >= 0 && new_x < n
					&& new_y >= 0 && new_y < n)
					data[new_x][new_y] = !data[new_x][new_y];
			}
		}
	};

	struct state_hasher {
		size_t operator()(const state& val) const noexcept {
			size_t res = 0;
			for (const auto& line : val.data) {
				size_t line_sum = 0;
				for (auto point : line)
					line_sum = (line_sum << 1) + point;
				res += line_sum;
			}
			return res;
		}
	};

	int test_case = 0;
	state init_state;
	priority_queue<state> open_set;
	unordered_set<state, state_hasher> close_set;

	// find the method turning parent to cur
	std::tuple<int, int, int> get_method(const state* cur, const state* parent) {
		vector<std::tuple<int, int>> diff;
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				if (cur->data[i][j] != parent->data[i][j])
					diff.emplace_back(std::make_tuple(i, j));
			}
		}
		if (diff.size() == 1) {
			auto& [x, y] = diff[0];
			if (x == 0 && y == 0) return std::make_tuple(x, y, 1);
			if (x == 0 && y == n - 1) return std::make_tuple(x, y, 0);
			if (x == n - 1 && y == 0) return std::make_tuple(x, y, 2);
			else return std::make_tuple(x, y, 3);
		}
		if (diff.size() == 2) {
			if (diff[0] > diff[1]) std::swap(diff[0], diff[1]);
			auto& [x1, y1] = diff[0];
			auto& [x2, y2] = diff[1];
			if (x1 == x2) {
				if (x1 == 0) return std::make_tuple(x1, y1, 0);
				else return std::make_tuple(x1, y1, 3);
			}
			if (y1 == y2) {
				if (y1 == 0) return std::make_tuple(x1, y1, 2);
				else return std::make_tuple(x1, y1, 3);
			}
		}
		else {
			/*
			* s == 1: (i-1,j), (i,j), (i,j+1);
			* s == 2: (i-1,j), (i,j-1), (i,j);
			* s == 3: (i,j-1), (i,j), (i+1,j);
			* s == 4: (i,j), (i,j+1), (i+1,j).
			*/
			std::sort(diff.begin(), diff.end());
			auto& [x1, y1] = diff[0];
			auto& [x2, y2] = diff[1];
			auto& [x3, y3] = diff[2];
			if (x1 != x2) {
				if (y1 == y2) return std::make_tuple(x2, y2, 0);
				else return std::make_tuple(x3, y3, 1);
			}
			else {
				if (y2 == y3) return std::make_tuple(x2, y2, 2);
				else return std::make_tuple(x1, y1, 3);
			}
		}
		return std::make_tuple(-1, -1, -1);
	}

	void print(const state* cur) {
		vector<std::tuple<int, int, int>> ans;
		while (*cur != init_state) {
			const state* parent = cur->parent;
			ans.emplace_back(get_method(cur, parent));
			cur = parent;
		}
		auto path = std::format("output/output{}.txt", test_case);
		std::ofstream fout(path);
		fout << ans.size() << '\n';
		for (auto method = ans.rbegin(); method != ans.rend(); ++method) {
			auto& [x, y, s] = *method;
			fout << std::format("{},{},{}\n", x, y, s);
		}
		fout.close();
	}
public:
	solve(int test_case) : test_case(test_case) {
		auto path = std::format("input/input{}.txt", test_case);
		std::ifstream fin(path);
		fin >> n;
		for (int i = 0; i < n; ++i) {
			vector<bool> line;
			for (int j = 0; j < n; ++j) {
				int input;
				fin >> input;
				line.push_back(input);
			}
			init_state.data.emplace_back(std::move(line));
		}
		fin.close();
	}

	void astar() {
		open_set.push(init_state);
		while (!open_set.empty()) {
			state cur = open_set.top();
			if (cur.loss() == 0) {
				print(&cur);
				return;
			}
			open_set.pop();
			auto [cur_iter, status] = close_set.insert(cur);
			++cur.g;
			for (int i = 0; i < n; ++i) {
				for (int j = 0; j < n; ++j) {
					for (int s = 0; s < 4; ++s) {
						cur.toward(i, j, s);
						if (close_set.find(cur) == close_set.end()) {
							cur.parent = &(*cur_iter);
							open_set.push(cur);
						}
						cur.toward(i, j, s);
					}
				}
			}
		}
	}
};

int main() {
	solve input0(0);
	input0.astar();
	return 0;
}
