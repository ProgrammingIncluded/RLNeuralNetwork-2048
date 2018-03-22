#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <unordered_set>

using namespace std;

struct Board{
	char tile[4][4] = {};
	
	static Board fromValue(unsigned long long v){
		Board ret;
		for (int i = 3; i >= 0; i--){
			for (int j = 3; j >= 0; j--){
				ret.tile[i][j] = v & 15;
				v >>= 4;
			}
		}
		return ret;
	}
	
	unsigned long long toValue() const{
		long long ret = 0;
		for (int i = 0; i < 4; i++){
			for (int j = 0; j < 4; j++){
				ret = ret * 16 + tile[i][j];
			}
		}
		return ret;
	}
	
	Board rotate() const{ // to the left
		Board ret;
		for (int i = 0; i < 4; i++){
			for (int j = 0; j < 4; j++){
				ret.tile[i][j] = tile[j][3 - i];
			}
		}
		return ret;
	}
	
	Board flip() const{
		Board ret;
		for (int i = 0; i < 4; i++){
			for (int j = 0; j < 4; j++){
				ret.tile[i][j] = tile[i][3 - j];
			}
		}
		return ret;
	}
	
	Board orient() const{
		unsigned long long v = toValue();
		Board r1 = rotate();
		v = min(v, r1.toValue());
		Board r2 = r1.rotate();
		v = min(v, r2.toValue());
		Board r3 = r2.rotate();
		v = min(v, r3.toValue());
		Board f = flip();
		v = min(v, f.toValue());
		Board fr1 = f.rotate();
		v = min(v, fr1.toValue());
		Board fr2 = fr1.rotate();
		v = min(v, fr2.toValue());
		Board fr3 = fr2.rotate();
		v = min(v, fr3.toValue());
		return Board::fromValue(v);
	}
	
	Board left(bool to_orient = false) const{
		Board ret;
		for (int i = 0; i < 4; i++){
			for (int j = 0, k = 0; j < 4; j++){
				if (tile[i][j] == 0) continue;
				if (ret.tile[i][k] == 0){
					ret.tile[i][k] = tile[i][j];
				}
				else if (ret.tile[i][k] == tile[i][j]){
					ret.tile[i][k]++;
					k++;
				} else {
					k++;
					ret.tile[i][k] = tile[i][j];
				}
			}
		}
		return to_orient ? ret.orient() : ret;
	}
	
	vector<Board> add(bool is4) const{
		vector<Board> ret;
		for (int i = 0; i < 4; i++){
			for (int j = 0; j < 4; j++){
				if (tile[i][j] == 0){
					Board temp = *this;
					temp.tile[i][j] = (is4 ? 2 : 1);
					ret.push_back(temp.orient());
				}
			}
		}
		return ret;
	}
	
	Board addRandom() const{
		Board ret = *this;
		int cnt = 0, r;
		for (int i = 0; i < 4; i++){
			for (int j = 0; j < 4; j++){
				if (tile[i][j] == 0){
					cnt++;
				}
			}
		}
		r = rand() % cnt;
		for (int i = 0; i < 4; i++){
			for (int j = 0; j < 4; j++){
				if (tile[i][j] == 0){
					if (r == 0) {
						ret.tile[i][j] = ((rand() % 10 == 0) ? 2 : 1);
						return ret;
					}
					r--;
				}
			}
		}
		return ret;
	}
	
	bool endGame() const{
		for (int i = 0; i < 4; i++){
			for (int j = 0; j < 4; j++){
				if (tile[i][j] == 0){
					return false;
				}
			}
		}
		return true;
	}
	
	char tileMax() const{
		char ret = 0;
		for (int i = 0; i < 4; i++){
			for (int j = 0; j < 4; j++){
				ret = max(ret, tile[i][j]);
			}
		}
		return ret;
	}
	
	double score() const{
		double ret = 0;
		for (int i = 0; i < 4; i++){
			for (int j = 0; j < 4; j++){
				double v = tile[i][j];
				ret += v * pow(2, v);
			}
		}
		return ret;
	}
	
	void print() const {
		for (int i = 0; i < 4; i++){
			for (int j = 0; j < 4; j++){
				cerr << (int) tile[i][j];
				if (j != 3) cerr << " ";
				else cerr << endl;
			}
		}
		cerr << endl;
	}
};

double sample(const Board cur){
	Board next = cur.addRandom();
	Board next2 = next.rotate();
	Board next3 = next2.rotate();
	Board next4 = next3.rotate();
	Board move = next.left();
	Board move2 = next2.left();
	Board move3 = next3.left();
	Board move4 = next4.left();
	bool prohibit = move.endGame();
	bool prohibit2 = move2.endGame();
	bool prohibit3 = move3.endGame();
	bool prohibit4 = move4.endGame();
	int cnt = (!prohibit) + (!prohibit2) + (!prohibit3) + (!prohibit4);
	if (cnt == 0) return cur.score();
	int r = rand() % cnt;
	if (!prohibit){
		if (r == 0) return sample(move);
		else r--;
	}
	if (!prohibit2){
		if (r == 0) return sample(move2);
		else r--;
	}
	if (!prohibit3){
		if (r == 0) return sample(move3);
		else r--;
	}
	if (!prohibit4){
		if (r == 0) return sample(move4);
		else r--;
	}
	return cur.score();
}

void MCTS(const Board cur, ofstream &fout){
	Board cur2 = cur.rotate();
	Board cur3 = cur2.rotate();
	Board cur4 = cur3.rotate();
	double maxV = cur.score(), max2 = cur.score();
	int argmax = -1, arg = 0;
	Board best;
	for (const Board &next: {cur.left(), cur2.left(), cur3.left(), cur4.left()}){
		if (next.endGame()) {
			arg++;
			continue;
		}
		double total = 0;
		for (int i = 0; i < 1000; i++){
			total += sample(next);
		}
		total /= 1000;
		if (total > maxV){
			//max2 = maxV;
			argmax = arg;
			maxV = total;
			best = next;
		}
		//else if (total > max2) max2 = total;
		arg++;
	} 
	cerr << "Expected Result: " << maxV << endl;
	cur.print();
	if (argmax == -1) return;
	fout << cur.toValue() << '\t' << argmax << '\t' << maxV << endl;
	MCTS(best.addRandom(), fout);
}

int main(int argc, char** argv){
	ofstream fout("mcts.txt");
	for (int i = 0; i < 20; i++) MCTS(Board().addRandom().addRandom(), fout);
	return 0;
}
