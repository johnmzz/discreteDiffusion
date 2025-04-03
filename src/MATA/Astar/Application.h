#ifndef _APPLICATION_H_
#define _APPLICATION_H_

#include <algorithm>
#include "Utility.h"
#include "Graph.h"

enum LB_Method { LSa, BMa, BMao };

struct State {
	State *parent;			// ��ǰ״̬�ĸ��׽ڵ�״̬
	State *pre_sibling;		// previous sibling
	ushort level;			// ��ǰ�Ĳ��
	ushort image;				// ƥ��ĵ㣿 
	ushort mapped_cost;			// ƥ��Ĵ���
	ushort lower_bound;			// ���ƵĴ���
	ushort cs_cnt; // cs_cnt stores how many children and right sibling depends on this state; this state can be removed if and only cs_cnt becomes 0
#ifdef _EXPAND_ALL_
	ushort *siblings; // remaining siblings			// ��ǰ��ʣ����ֵܽڵ�
	ushort siblings_n;
#endif
};

class Application {
	//private:
public:
	ui q_n;			// q ��ʾ��һ��ͼ ����С��ͼ��
	ui *q_starts, *q_edges;
	ui *q_vlabels, *q_elabels;

	ui g_n;			// g ��ʾ�ڶ���ͼ ���ϴ��ͼ��
	ui *g_starts, *g_edges;
	ui *g_vlabels, *g_elabels;   // g_vlabels: �ڶ���ͼ�У�ÿ���ڵ�ı�ǩ

	ui vlabels_n, elabels_n;

	ui beam_size;
	ui verify_upper_bound;
	ui upper_bound;

	LB_Method lb_method;

	bool q_g_swapped;		// �ж� ��һ����ͼ�Ƿ񽻻�

	
	ui search_n;
	ui search_n_for_IS;

	char *visX, *visY;		// visX: ���q�е���Щ�ڵ��Ѿ���ƥ��
	int *mx, *my;
	ui *BX;
	ui *candidates;
	ui *queue, *prev;

	std::vector<State *> states_memory;				// ���״̬��
	std::vector<State *> states_pool;				// ״̬�أ���ʼ��1024��״̬����
	ui states_pool_n;								// ״̬����ʣ���״̬����

	std::vector<ushort *> siblings_memory;			// ��� �ֵܽڵ��
	std::vector<ushort *> siblings_pool;			// �ֵܽڵ��
	ui siblings_pool_n;								// �ֵܽڵ����ʣ��ڵ�ĸ���

	int *elabels_map, *vlabels_map;					// �������ᵽ���������ݽṹ��ע����g��δƥ�����ͼ - q��δƥ�����ͼ���ڵ�u���ڴ���ͼ�У�
	short *elabels_matrix;
	uchar *q_matrix;		// TODO:

	ushort *visited_siblings; // these two arrays only used in DFS
	ushort *visited_siblings_n;

	/***** for bipartite matching based lower bounds ****/
	ui *cost;
	int *lx, *ly, *slack, *slackmy;

	std::pair<int, int> *children;

	long long search_space;
	
	int *MO;									// ƥ��Ľڵ�˳��matching order. ui * MO. 
	int** matched_nodes;						// topK matched nodes for each node
	int k;										// topK
	bool **in_matched_nodes;					// ����matched_nodes��������ʾÿ���ڵ㣬i,j �Ƿ�ƥ��

public:
	Application(ui _beam_size, const char *lower_bound);
	~Application();

	long long get_search_space() { return search_space; }

	void init(const std::vector<std::pair<int, int> > &g_v, const std::vector<std::pair<std::pair<int, int>, int> > &g_e, const std::vector<std::pair<int, int> > &q_v, const std::vector<std::pair<std::pair<int, int>, int> > &q_e);
	void init(const Graph *g, const Graph *q, int* ub_map, int* MO, int**matched_nodes, int k);
	ui DFS(State *node = NULL);
	ui AStar(std::vector<std::pair<ui, ui> > *mapping_ptr = NULL, int *lb_ptr = NULL);
	ui compute_ged_of_BX();
	int* compute_operation_of_BX();
	bool get_mapping(std::vector<std::pair<ui, ui> > &mapping, ui n);

public:
	void preprocess();

	void add_to_pool(State *st);
	void add_to_heap(State *st, ui &heap_n, std::vector<State*> &heap);
	State* get_a_new_state_node();
	void put_a_state_to_pool(State *st);
	ushort* get_a_new_siblings_node();
	void put_a_sibling_to_pool(ushort *sibling);
	void verify_induced_ged(State *now);
	void verify_LS_lower_bound(State *now);
	ui relabel(ui len1, ui *array1, ui len2, ui *array2);
	ui search_index(ui val, std::vector<ui> &array, ui array_len);

	void compute_mapping_order();
	void generate_best_extension(State *parent, State *now); // compute the best child of a state
	void compute_mapped_cost_and_upper_bound(State *now, ui n, ui *candidates, int *mapping);
	void construct_sibling(State *pre_sibling, State *now); // compute the best ungenerated sibling of a state
	void extend_to_full_mapping(State *parent, State *now);
	void compute_mapped_cost(State *now);


	void compute_best_extension_LSa(State *now, ui candidate_n, ui *candidates, ui pre_siblings);
	void compute_best_extension_BM(char anchor_aware, State *now, ui candidate_n, ui *candidates, ui pre_siblings, char no_siblings, char IS = 0);
	void compute_best_extension_BMa(char anchor_aware, State *now, ui n, ui *candidates, ui pre_siblings);

	ui Hungarian(char initialization, ui n, ui *cost); // minimum cost bipartite matching
	void heap_top_down(ui idx, ui heap_n, std::vector<std::pair<double, int> > &heap, ui *pos);
	void heap_bottom_up(ui idx, std::vector<std::pair<double, int> > &heap, ui *pos);
	void heap_top_down(ui idx, ui heap_n, std::vector<State*> &heap);
	void heap_bottom_up(ui idx, std::vector<State*> &heap);

	ui compute_ub_map(int *ub_map);
	void print_heap(ui heap_n, std::vector<State*> &heap);
	int find_first_match(std::pair<int, int> *children, int size, bool *cur_matched_nodes);
};

#endif
