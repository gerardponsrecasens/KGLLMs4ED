# Script to create and store final graphs for DBpedia. It requires util functions
import os
import json
import pickle
from utils import *

os.makedirs('./subgraphs_db')

datasets = ['wiki','aqu','ace2004','cweb','KORE50','msn','oke15','oke16','reu','RSS ']

for data_name in datasets:

    os.makedirs('./subgraphs_db/'+data_name+'_db_chatel')

    # Open the dataset
    with open(r'./data/'+data_name+'_chatel.jsonl', 'r', encoding='utf-8') as file:
        for line in file: # Read each mention
            data = json.loads(line) #Import line as json
            id = data['id']
            candidates = data['candidates']
            answer = data['answer']
            len_candidates = len(candidates)
            answer_in_candidates = answer in candidates


            if len_candidates ==0: #We do not consider the cases if the candidate set is empty
                continue

            # Look if answer is in the KB, otherwise we are not inKB
            check_answer = 'http://dbpedia.org/resource/' + answer.replace(' ','_').replace('"','')
            a_types = type_query(check_answer)
            if len(a_types) ==0:
                a_types = type_query(check_answer, ns='http://www.w3.org/2002/07/owl#')
                if len(a_types) == 0: #We do not record it if it is not in KB
                    continue

            candidates = [i.replace(' ','_') for i in candidates]


            ################## LOAD THE CLASS TAXONOMY ###############################


            with open(r'./data/dbo.pkl', 'rb') as handle:
                G = pickle.load(handle)
            
            ################## LINK CANDIDATES TO CLASSES ###########################

            for candidate in candidates:

                    # Find the types of the candidate
                    types = type_query('http://dbpedia.org/resource/' + candidate.replace('"',''))

                    if len(types) == 0: # If there are no associated candidates in dbo, look for OWL
                        types = type_query('http://dbpedia.org/resource/' + candidate.replace('"',''), ns= 'http://www.w3.org/2002/07/owl#')

                    if len(types) == 0: # If there are no types, the entity is not in the KB -> no inKB
                        continue
                    
                    ################## IGNORE NO DIRECT PATHS ###########################
                    for current_type in types:
                        if G.has_node(current_type): # Some types are not in the hierarchy because they are 'sameAs'
                            successors = list(G.successors(current_type)) #If the type has childs that are also candidates, it can be omitted
                            direct = True
                            for other_types in types:
                                if other_types in successors:
                                    direct = False
                            
                            if direct:
                                G.add_edge(current_type,candidate) # Add the edge
                        
            
            
            ################## REMOVE UNRELATED CLASSES #########################

            out_degree_dict = dict(G.out_degree())
            to_remove = [key for key, value in out_degree_dict.items() if (value == 0) and (key not in candidates)]  

            while len(to_remove) !=0:
                for node in to_remove:
                    G.remove_node(node)
                out_degree_dict = dict(G.out_degree())
                to_remove = [key for key, value in out_degree_dict.items() if (value == 0) and (key not in candidates)]
                    
            out_degree_dict = dict(G.out_degree())
            candidates = [key for key, value in out_degree_dict.items() if value == 0]


            ######################## REMOVE SELF POINTING EDGES ##########################

            G = remove_self_pointing_edges(G)            

            ################## REMOVE INTERMEDIATE NODES #########################
            
            nodes_1_child = [node for node in G.nodes() if (node!='owl#Thing') and (len(list(G.successors(node)))==1) and (len(list(G.successors(list(G.successors(node))[0])))>0)]

            while (len(nodes_1_child)) !=0:
                for node in nodes_1_child:
                    child = list(G.successors(node))[0]
                    for pred in list(G.predecessors(node)):
                        G.add_edge(pred,child)
                    G.remove_node(node)


                nodes_1_child = [node for node in G.nodes() if (node!='owl#Thing') and (len(list(G.successors(node)))==1) and (len(list(G.successors(list(G.successors(node))[0])))>0)]
            
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
            
            with open('./subgraphs_db/'+data_name+'_db_chatel/'+str(id)+'.pkl', 'wb') as f:
                pickle.dump([G,candidates], f)