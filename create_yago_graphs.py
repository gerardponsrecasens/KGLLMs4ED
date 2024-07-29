# Script to create and store final graphs for DBpedia. It requires util functions
import os
import json
from utils import *
import pickle


os.makedirs('./subgraphs_yago')

datasets = ['wiki','aqu','ace2004','cweb','KORE50','msn','oke15','oke16','reu','RSS ']

for data_name in datasets:

    os.makedirs('./subgraphs_yago/'+data_name+'_yago_chatel')

    # Open dataset
    with open(r'./data/'+data_name+'_chatel.jsonl', 'r', encoding='utf-8') as file:
        for line in file: # Read each mention
            data = json.loads(line) #Import line as json
            id = data['id']
            candidates = data['candidates']
            answer = data['answer']
            len_candidates = len(candidates)
            answer_in_candidates = answer in candidates

            candidates = yago_input(candidates, ns=True)

            if len_candidates == 0: 
                continue
            
            check_answer = yago_input([answer], ns = True)[0]
            types = get_subclasses(check_answer,'rdf:type')
            if len(types) == 0:
                types = get_subclasses(check_answer,'rdfs:subClassOf')
                if len(types) == 0:
                    continue #Not in KB scenario

            ################### LINK TO ONLY RELATED CLASSES ##################################

            G = build_graph(candidates)

             ################## REMOVE SELF POINTING EDGES #############################

            G = remove_self_pointing_edges(G)

            ################## REMOVE INTERMEDIATE NODES ######################### 

            nodes_1_child = [node for node in G.nodes() if (node!='Thing') and (len(list(G.successors(node)))==1) and (len(list(G.successors(list(G.successors(node))[0])))>0)]

            while (len(nodes_1_child)) !=0:
                for node in nodes_1_child:
                    child = list(G.successors(node))[0]
                    for pred in list(G.predecessors(node)):
                        G.add_edge(pred,child)
                    G.remove_node(node)


                nodes_1_child = [node for node in G.nodes() if (node!='Thing') and (len(list(G.successors(node)))==1) and (len(list(G.successors(list(G.successors(node))[0])))>0)]

            ################## FIX CANDIDATES THAT ARE ALSO LEAVES #########################

            for candidate in candidates:
                if G.has_node(candidate):
                    childs = list(G.successors(candidate))
                    if len(childs) !=0:
                        parents = list(G.predecessors(candidate))
                        for child in childs:
                            G.remove_edge(candidate,child)
                            for parent in parents:
                                G.add_edge(parent,child)


            out_degree_dict = dict(G.out_degree())
            candidates = [key for key, value in out_degree_dict.items() if value == 0]

            with open('./subgraphs_yago/'+data_name+'_yago_chatel/'+str(id)+'.pkl', 'wb') as f:
                    pickle.dump([G,candidates], f)