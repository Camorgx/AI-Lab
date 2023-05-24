#include <format>
#include <fstream>
#include <iostream>
#include <numeric> // std::accumulate
#include <queue>
#include <tuple>
#include <unordered_set>
#include <vector>

// used std data structrue
using std::priority_queue;
using std::unordered_set;
using std::vector;

class solve {
	int test_case = 0;
	int staff_num = 0, days_num = 0, shifts_num = 0;
	int least_dis = 0;
	vector<vector<int>> shifts;
	using size_pos_tuple = std::tuple<size_t, int, int>; // domain size, day, slot
	using domain_size_heap = priority_queue<size_pos_tuple, 
		vector<size_pos_tuple>, std::greater<size_pos_tuple>>;
	domain_size_heap avaiable;
	// in day d, slot s, staff that is available
	vector<vector<unordered_set<int>>> domain;
	// times that a staff is distributed
	vector<int> dis_count; 
	// slot that nobody would like to work
	vector<std::tuple<int, int>> nobody_slot;

	bool csp_check() const {
		for (int count : dis_count) {
			if (count < least_dis)
				return false;
		}
		return true;
	}

	void print() {
		auto path = std::format("output/output{}.txt", test_case);
		std::ofstream fout(path);
		for (const auto& line : shifts) 
			for (int i = 0; i < shifts_num; ++i) 
				fout << line[i] + 1 << (i != shifts_num - 1) ? ',' : '\n';
		for (const auto& [day, slot] : nobody_slot)
			domain[day][slot].clear();
		int unmet_req_cnt = 0;
		for (int i = 0; i < days_num; ++i) {
			vector<int> unmet_req(staff_num);
			for (int j = 0; j < shifts_num; ++j) {
				unmet_req[shifts[i][j]] |=
					int(domain[i][j].find(shifts[i][j]) == domain[i][j].end());
			}
			unmet_req_cnt += std::accumulate(unmet_req.begin(), unmet_req.end(), 0);
		}
		fout << staff_num * days_num - unmet_req_cnt << std::endl;
		fout.close();
	}
public:
	solve(int test_case) : test_case(test_case) {
		auto path = std::format("input/input{}.txt", test_case);
		std::ifstream fin(path);
		char ch; // used to get ','
		fin >> staff_num >> ch >> days_num >> ch >> shifts_num;
		least_dis = days_num * shifts_num / staff_num;
		dis_count.resize(staff_num);
		for (int i = 0; i < days_num; ++i) {
			domain.emplace_back(shifts_num);
			shifts.emplace_back(shifts_num);
		}
		for (int i = 0; i < staff_num; ++i) {
			for (int j = 0; j < days_num; ++j) {
				for (int k = 0; k < shifts_num; ++k) {
					int ok;
					fin >> ok;
					if (k != shifts_num - 1) fin >> ch;
					if (ok) domain[j][k].insert(i);
				}
			}
		}
		fin.close();
		for (int i = 0; i < days_num; ++i) 
			for (int j = 0; j < shifts_num; ++j) {
				size_t candi_cnt = domain[i][j].size();
				if (candi_cnt == 0) {
					candi_cnt = staff_num;
					nobody_slot.emplace_back(std::make_tuple(i, j));
					for (int s = 0; s < staff_num; ++s)
						domain[i][j].insert(s);
				}
				avaiable.push(std::make_tuple(candi_cnt, i, j));
			}
	}

	bool csp_search() {
		if (avaiable.empty()) {
			if (csp_check()) {
				print();
				return true;
			}
			else return false;
		}
		auto [d_size, day, slot] = avaiable.top();
		avaiable.pop();
		for (int candidate : domain[day][slot]) {
			int origin = shifts[day][slot];
			shifts[day][slot] = candidate;
			size_t pre_erase_cnt = 0, next_erase_cnt = 0;
			if (slot > 0) 
				pre_erase_cnt = domain[day][slot - 1].erase(candidate);
			if (slot < shifts_num - 1) 
				next_erase_cnt = domain[day][slot + 1].erase(candidate);
			++dis_count[candidate];
			if (csp_search()) return true;
			--dis_count[candidate];
			if (pre_erase_cnt) domain[day][slot - 1].insert(candidate);
			if (next_erase_cnt) domain[day][slot + 1].insert(candidate);
			shifts[day][slot] = origin;
		}
		avaiable.push(std::make_tuple(d_size, day, slot));
		return false;
	}
};

int main() {
	solve solver(0);
	if (solver.csp_search())
		std::cout << "Valid schedule found." << std::endl;
	else std::cout << "No valid schedule found." << std::endl;
	return 0;
}
