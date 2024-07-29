'''
This code orchestrates the pipeline for the obtention of the LLM results of the ED task over the 
YAGO KB.
'''

import pickle
import json
import csv
import os

from mistralai.client import MistralClient
from utils import *
from openai import OpenAI
import networkx as nx

dataset_list = ['wiki','aqu','ace2004','cweb','KORE50','msn','oke15','oke16','reu','RSS ']
os.makedirs('./results/yago') 

for dataset_name in dataset_list:

    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print(dataset_name)
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')


    # Experiment variables
    llm_provider = 'openai' #llmstudio, openai, mistral
    model_name = 'gpt35' #mistral-large, mistral-small, gpt35, mistral7B
    ontology = 'yago' #db
    candidateSet = 'chatel' # chatel
    dataset = dataset_name
    model = "gpt-3.5-turbo-1106" # 'mistral-large-latest' 
    check = True # For LLMs with a lot of erroneous responses

    experiment_name = dataset + '_pipeline_'+model_name+'_'+ontology+'_'+candidateSet
    dataset_file = r'./data/'+dataset+'_'+candidateSet+'.jsonl'

    # LLM settings

    if llm_provider == 'openai':
        client = OpenAI(api_key="you_openai_key")
    elif llm_provider == 'llmstudio':
        client = OpenAI(base_url="you_local_host_ID", api_key="not-needed")
    elif llm_provider == 'mistral':
        client = MistralClient(api_key='your_mistral_key')

    # Load first line

    with open(dataset_file, 'r', encoding='utf-8') as file:

        with open('./results/yago/'+experiment_name+'.csv', 'a', newline='', encoding='utf-8') as csvfile:

            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['id', 'candidate_length', 'in_candidates', 'response', 'answer'])

            for line in file: # Read each mention
                data = json.loads(line) #Import line as json
                id = data['id']
                full_candidates = data['candidates']
                mention = data['mention']
                text = data['input']
                answer = data['answer']
                len_candidates = len(full_candidates)
                answer_in_candidates = answer in full_candidates
                

                # Look if we have created the pickle for them:
                if os.path.isfile(r'./subgraphs_yago/'+dataset+'_'+ontology+'_'+candidateSet+'/'+str(id)+'.pkl'):
                    with open(r'./subgraphs_yago/'+dataset+'_'+ontology+'_'+candidateSet+'/'+str(id)+'.pkl', 'rb') as file:
                        G,candidates = pickle.load(file)
                        len_candidates = len(candidates)
                        yago_answer = answer.replace(' ','_')
                        answer_in_candidates = yago_answer in candidates
    
                else: # Not inKB
                    csvwriter.writerow([id, len_candidates, answer_in_candidates, 'notInKB', 'notInKB'])
                    continue

                if not answer_in_candidates:
                    csvwriter.writerow([id, len_candidates, answer_in_candidates, 'notInC', answer])
                    continue
                
                elif len(candidates)==1:
                    csvwriter.writerow([id, len_candidates, answer_in_candidates, clean(candidates)[0], answer])
                    continue


                iteration = 1

                # Keep the loop while there is more than one candidate
                while len(candidates) > 1:

                    candidate_lca = LCA_YAGO(G, candidates)
                    childs = list(G.successors(candidate_lca))
                    config = child_config(G, candidate_lca)


                    ###################################### CASE 1 #####################################################

                    if config == 1:

                        prompt = case_1_query(childs,text, mention)
                        llm = json.loads(get_response(prompt, client, provider=llm_provider, model=model))['category']

                        # We start in the case that the LLM says none of the categories match our proposed experiment
                        if llm == 'None':
                            prompt = case_none_query(full_candidates,text,mention)
                            llm = json.loads(get_response(prompt, client, provider=llm_provider, model=model))['entity']
                            candidates = [llm]
                            iteration += 1
                        
                        elif llm in childs: #The LLM selected a category
                            childs.remove(llm)
                            G = prune_graph_YAGO(G,llm)
                            out_degree_dict = dict(G.out_degree())
                            candidates = [key for key, value in out_degree_dict.items() if value == 0]
                        
                        else:
                            match = exact_match(llm,childs,client,llm_provider, model, similarity=False)
                            if match in childs:
                                G = prune_graph_YAGO(G,match)
                                out_degree_dict = dict(G.out_degree())
                                candidates = [key for key, value in out_degree_dict.items() if value == 0]
                            else:
                                candidates = ['BadLLM']

                    ###################################### CASE 2 #####################################################
                                                        
                    elif config == 2:

                        prompt = case_2_query(candidates,text, mention)
                        llm = json.loads(get_response(prompt,client, provider=llm_provider, model=model))['entity']


                        if llm == 'None':
                            prompt = case_none_query(full_candidates,text,mention)
                            llm = json.loads(get_response(prompt, client, provider=llm_provider, model=model))['entity']
                            candidates = [llm]
                            iteration += 1
                        
                        elif llm in candidates:
                            candidates = [llm]
            
                        else:
                            match = exact_match(llm,candidates,client,llm_provider, model, similarity=False)
                            if match in candidates:
                                candidates = [match]
                            candidates = ['BadLLM']
                    
                    ###################################### CASE 3 #####################################################
                    else:
                        ent = [] 
                        subclasses = [] 
                        leaves = [] 
                        for child in childs:
                            if len(list(G.successors(child))) == 0:
                                leaves.append(child)
                                # A child may be an entity that is also a child of one of the other remaining classes
                                set_pred = set(list(G.predecessors(child)))
                                set_childs = set(childs)
                                if not set_pred.intersection(set_childs):
                                    ent.append(child)
                            else:
                                subclasses.append(child)

                        # We ask the query
                        prompt = case_3_query(subclasses,text, mention)
                        llm = json.loads(get_response(prompt,client, provider=llm_provider, model=model))['category']
                        
                        good_output = True

                        if llm not in subclasses+['Other']:
                            llm = exact_match(llm,subclasses+['Other'],client,llm_provider, model, similarity=False)

                        if llm == 'Other':
                            del G
                            G = nx.DiGraph()
                            if len(ent) !=0:
                                for entity in ent:
                                    G.add_edge('Thing',entity)
                            else:
                                for entity in leaves:
                                    G.add_edge('Thing',entity)
                        else:
                            if llm in subclasses:
                                subclasses.remove(llm)
                                G = prune_graph_YAGO(G,llm)
                            else:
                                candidates = ['BadLLM']
                                good_output = False

                        if good_output:
                            out_degree_dict = dict(G.out_degree())
                            candidates = [key for key, value in out_degree_dict.items() if value == 0]

                    
                    # CHECK 
                    if check and ((config != 2 and len(candidates) == 1) or candidates[0] == 'BadLLM'):
                        candidates = check_errors(candidates, full_candidates,text, mention, prompt,client,llm_provider, model)
                        iteration += 1

                    iteration += 1


                

                csvwriter.writerow([id, len_candidates, answer_in_candidates, clean(candidates)[0], answer])

