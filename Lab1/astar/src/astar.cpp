#include <chrono>
#include <cmath>
#include <format>
#include <fstream>
#include <iostream>
#include <queue>
#include <tuple>
#include <unordered_set>
#include <vector>

// used std data structrue
using std::priority_queue;
using std::unordered_set;
using std::vector;

int n; // size of the map
int test_case;

class solve {
	class state {
	public:
		int g = 0;
		int evaluate = 0;
		const state* parent = nullptr;
		vector<vector<bool>> data;

		int f() const { return g + h(); }

		// count the number of 1
		int loss() const {
			int res = 0;
			for (const auto& line : data)
				for (const auto& point : line)
					res += int(point);
			return res;
		}

		size_t get_8_connect_size(int x, int y, vector<vector<bool>>& vis) const {
			if (vis[x][y] || !data[x][y]) return 0;
			vis[x][y] = true;
			constexpr int x_trans[] = { -1, 0, 1, -1, 1, -1, 0, 1 };
			constexpr int y_trans[] = { -1, -1, -1, 0, 0, 1, 1, 1 };
			size_t res = 1;
			for (int i = 0; i < 8; ++i) {
				int new_x = x + x_trans[i];
				int new_y = y + y_trans[i];
				if (new_x < 0 || new_x >= n || new_y < 0 || new_y >= n)
					continue;
				if (vis[new_x][new_y]) continue;
				res += get_8_connect_size(new_x, new_y, vis);
			}
			return res;
		}

		int h() const {
			vector<vector<bool>> vis(n, vector<bool>(n));
			double res = 0;
			for (int i = 0; i < n; ++i) {
				for (int j = 0; j < n; ++j) {
					if (data[i][j] && !vis[i][j]) {
						res += std::ceil(get_8_connect_size(i, j, vis) / 3.0);
					}
				}
			}
			int ret = static_cast<int>(res);
			if ((loss() & 1) != (ret & 1)) ++ret;
			return ret;
		}

		// std::priority_queue defaults to a large root heap  
		bool operator<(const state& ano) const {
			return evaluate > ano.evaluate;
		}

		bool operator==(const state& ano) const {
			return data == ano.data;
		}

		// turn with method s at point (x, y)
		bool toward(int x, int y, int s) {
			if (x < 0 || x >= n || y < 0 || y >= n)
				return false;
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
			bool valid = true;
			for (int i = 0; i < 3; ++i) {
				int new_x = x + x_trans[s][i];
				int new_y = y + y_trans[s][i];
				if (new_x < 0 || new_x >= n || new_y < 0 || new_y >= n)
					valid = false;
			}
			if (valid) {
				for (int i = 0; i < 3; ++i) {
					int new_x = x + x_trans[s][i];
					int new_y = y + y_trans[s][i];
					data[new_x][new_y] = !data[new_x][new_y];
				}
			}
			return valid;
		}

		std::string to_string() const {
			std::string res;
			for (const auto& line : data) {
				for (const auto& point : line)
					res += std::format("{} ", int(point));
				res += '\n';
			}
			return res;
		}
	};

	// hash function for state, used in std::unordered_map
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
			fout << std::format("{},{},{}\n", x, y, s + 1);
		}
		fout.close();
	}
public:
	solve(int t_case) {
		test_case = t_case;
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
		init_state.evaluate = init_state.f();
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
			int i = 0, j = 0;
			for (i = 0; i < n; ++i)
				for (j = 0; j < n; ++j)
					if (cur.data[i][j]) goto FOUND;
			FOUND:
			const std::tuple<int, int, int> possibility[] = {
				{i, j, 0}, {i, j, 1}, {i, j, 2}, {i, j, 3},
				{i, j + 1, 1}, {i, j + 1, 2},
				{i + 1, j, 0}, {i + 1, j, 1},
				{i - 1, j, 3}, 
				{i, j - 1, 3}, {i, j - 1, 0},
			};
			for (const auto& [x, y, method] : possibility) {
				if (cur.toward(x, y, method)) {
					if (close_set.find(cur) == close_set.end()) {
						cur.evaluate = cur.f();
						cur.parent = &(*cur_iter);
						open_set.push(cur);
					}
					cur.toward(x, y, method);
				}
			}
		}
	}

	void verify() const {
		state init(init_state);
		auto path = std::format("output/output{}.txt", test_case);
		std::ifstream fin(path);
		int step;
		fin >> step;
		for (int i = 0; i < step; ++i) {
			int x, y, s;
			char ch;
			fin >> x >> ch >> y >> ch >> s;
			init.toward(x, y, s - 1);
		}
		if (init.loss() == 0)
			std::cout << std::format("Test case {} passed.", test_case) << std::endl;
		else {
			std::cout << std::format("Test case {} failed.\n", test_case);
			std::cout << "Final state:\n";
			std::cout << init.to_string() << std::endl;
		}
	}
};

int main() {
	for (int i = 0; i < 10; ++i) {
		solve solver(i);
		auto start = std::chrono::steady_clock::now();
		solver.astar();
		auto finish = std::chrono::steady_clock::now();
		// get duration by ms
		std::chrono::duration<double, std::milli> duration = finish - start;
		std::cout << std::format("Computation of test {} finished in {}.\n", 
			i, duration);
		solver.verify();
		std::cout << std::endl;
	}
	return 0;
}
