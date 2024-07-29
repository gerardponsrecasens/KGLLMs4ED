from mistralai.models.chat_completion import ChatMessage
from SPARQLWrapper import SPARQLWrapper, JSON
import xml.etree.ElementTree as ET
import networkx as nx
import wikipediaapi
import Levenshtein
import requests
import json
import re


############################################################################################
#                                   CREATE GRAPHS                                          #
############################################################################################

def get_subclasses(entity,relation):

    '''
    Function that given an entity uri, returns what it is subclass of in YAGO
    :param entity: uri of a class/entity in YAGO (e.g., http://schema.org/Organization)
    :param relation: either rdfs:subClassOf or rdf:type
    '''
    # SPARQL endpoint URI
    endpoint_url = "https://yago-knowledge.org/sparql/query"

    # SPARQL query
    sparql_query = f"""
    PREFIX schema: <http://schema.org/>
    PREFIX yago: <http://yago-knowledge.org/resource/>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT * WHERE {{
    <{entity}> {relation} ?class.
    }}
    """

    # Set headers and data for the POST request
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }

    data = {
        "query": sparql_query,
    }

    # Send the POST request to the SPARQL endpoint
    response = requests.post(endpoint_url, headers=headers, data=data)

    uris = []
    # Check if the request was successful
    if response.ok:
        # Access the response as text
        xml_response = response.text
        # Parse the XML string
        root = ET.fromstring(xml_response)

        # Define the namespace
        namespace = {'ns': 'http://www.w3.org/2005/sparql-results#'}

        # Use XPath to find all uri elements
        uri_elements = root.findall(".//ns:uri", namespaces=namespace)

        # Extract and print the URIs
        uris = [uri.text for uri in uri_elements]
    
    return uris

def type_query(entity, ns = "http://dbpedia.org/ontology/"):
  
  '''
  Function to get the types of an entity in DBPedia
  :param entity: the name of the entity in DBPedia URL
  :param ns: name space (either the DBPedia or the OWL)
  '''

  entity = '<'+entity+'>'
  sparql = SPARQLWrapper("http://dbpedia.org/sparql")

  # Construct the query with string formatting
  query = """
  SELECT DISTINCT ?type
  WHERE {
    %s a ?type .
    FILTER(STRSTARTS(str(?type), "%s") && !regex(str(?subclass), ".*[\\u0600-\\u06FF].*"))
  }
  """ % (entity, ns)

  # Set the SPARQL query
  sparql.setQuery(query)

  # Set the format of the response (in this case, JSON)
  sparql.setReturnFormat(JSON)

  # Execute the query and parse the results
  results = sparql.query().convert()

  types = []
  for result in results['results']['bindings']:
    types.append(result['type']['value'].split('/')[-1])

  return types



def build_graph(entities):
    '''
    Given a list of entities (yago urls), it builds a graph with all the hierarchy of subclasses from the entities up to schema:Thing
    It does not use the namespaces for the names, removes the _Q+numbers of some entities and also changes special characters.
    :param entities: list of entities 
    '''
    tree = []
    classes = []
    pattern = re.compile(r'_Q\d+$')

    for entity in entities:

        # 1 Look for type and subclasses
        types = get_subclasses(entity,'rdf:type')
        if 'http://www.w3.org/2000/01/rdf-schema#Class' in types:
            types.remove('http://www.w3.org/2000/01/rdf-schema#Class')
        subclasses = get_subclasses(entity,'rdfs:subClassOf')

        # Add the nodes to the list
        for t in types:
            tree.append((entity.split('/')[-1],'is_a',t.split('/')[-1]))
        
        for sc in subclasses:
            tree.append((entity.split('/')[-1],'is_a',sc.split('/')[-1]))

        types.extend(subclasses) #Merge both lists

        while len(types) != 0:
            new_class = types.pop(0)
            if new_class not in classes: #We keep track of the inspected classess, so we do not look twice the same node
                classes.append(new_class) 

                new_parentClasses = get_subclasses(new_class,'rdfs:subClassOf') #Get the parrent classess of the subclass

                for npc in new_parentClasses:
                    types.append(npc) #Put it in the list so we look for its parents later in the loop
                    tree.append((new_class.split('/')[-1],'is_a',npc.split('/')[-1])) #Add the node in the graph


    # Create Graph
    G = nx.DiGraph()

    # Add edges to the graph based on the list data
    for entry in tree:
        child, relationship, parent = entry
        child = pattern.sub('', child) #Remove the _Q+numbers
        parent = pattern.sub('', parent) #Remove the _Q+numbers
        child = wikipedia_input([child])[0] # Remove special characters
        parent = wikipedia_input([parent])[0] #Remove special characters

        G.add_edge(parent, child)
    
    return G

def remove_self_pointing_edges(G):
    edges_to_remove = []
    for u, v in G.edges():
        if u == v:
            edges_to_remove.append((u, v))
    for u, v in edges_to_remove:
        G.remove_edge(u, v)
    
    return G


############################################################################################
#                                   PRUNE GRAPHS                                           #
############################################################################################


def LCA(G,lcas):
    list_of_lists = []

    for el in lcas:
        list_of_lists.append(list(nx.ancestors(G, el)))
    sets = [set(inner_list) for inner_list in list_of_lists] 
    common_elements = sets[0]
  
    for s in sets[1:]:
        common_elements.intersection_update(s)
    common = list(common_elements)

    if len(common) == 1:
        return 'owl#Thing'
    else:
        common.remove('owl#Thing')
        distance = []
        for el in common:
            distance.append(nx.shortest_path_length(G, source='owl#Thing', target=el))
        
        return  common[distance.index(max(distance))]

def LCA_YAGO(G,lcas):
    list_of_lists = []

    for el in lcas:
        list_of_lists.append(list(nx.ancestors(G, el)))
    sets = [set(inner_list) for inner_list in list_of_lists] 
    common_elements = sets[0]
  
    for s in sets[1:]:
        common_elements.intersection_update(s)
    common = list(common_elements)
    print('common_elements_in_LCA:',common)
    if len(common) == 1:
        return 'Thing'
    else:
        common.remove('Thing')
        distance = []
        for el in common:
            distance.append(nx.shortest_path_length(G, source='Thing', target=el))
        
        return  common[distance.index(max(distance))]
    
def child_config(G,node):
    '''
    Given a node, we assess if all its children are classes (1), if all are leaves (2) or if there is a mix (3)
    :param G: graph (networkx object) we are working with
    :param node: node to assess childs
    '''
    childs = list(G.successors(node))
    n_childs = len(childs)
    out_edges = list(dict(G.out_degree(childs)).values())
    num_leaves = sum([i == 0 for i in out_edges])

    if num_leaves == n_childs:
        return 2
    elif num_leaves == 0:
        return 1
    else:
        return 3
    
def prune_graph(G,elements, owl_thing = 'Thing'):
    '''
    Function to prune the graph by deleting the elements and their successors without other parents
    :param G: graph (networkx object) to prune
    :param elements: elements to be removed from the graph
    :param owl_thing: how the top element owl#Thing is stored in the KB
    '''

    extra = [] #Class nodes that may be childless indirectly
    for element in elements:
        predecessors = list(G.predecessors(element))
        G.remove_node(element)
        for pred in predecessors:
          if len(list(G.successors(pred)))==0:
            extra.append(pred)
        

    extra = list(set(extra))

    while len(extra) !=0:
        new_extra = []
        for element in extra:
            predecessors = list(G.predecessors(element))
            G.remove_node(element)
            for pred in predecessors:
                if len(list(G.successors(pred)))==0:
                    new_extra.append(pred)
        extra = list(set(new_extra))


    in_degree_dict = dict(G.in_degree())
    keys_in_degree_0 = [key for key, value in in_degree_dict.items() if value == 0]
    
    if owl_thing in keys_in_degree_0:
        keys_in_degree_0.remove(owl_thing)

    while len(keys_in_degree_0):
        for key in keys_in_degree_0:
            G.remove_node(key)
        in_degree_dict = dict(G.in_degree())
        keys_in_degree_0 = [key for key, value in in_degree_dict.items() if value == 0]
        keys_in_degree_0.remove(owl_thing)

    return G

def prune_graph_YAGO(G, winner,owl_thing = 'Thing'):
    '''
    Function to prune the graph by deleting the elements and their successors without other parents
    :param G: graph (networkx object) to prune
    :param elements: elements to be removed from the graph
    :param winner: the element that was selected
    :param owl_thing: how the top element owl#Thing is stored in the KB
    '''
    out_degree_dict = dict(G.out_degree())
    candidates = [key for key, value in out_degree_dict.items() if value == 0]

    nodes = list(G.nodes())
    desc = list(nx.descendants(G,winner))
    desc.append('Thing')
    desc.append(winner)
    G.add_edge('Thing',winner)

    difference = [item for item in nodes if item not in desc]

    for node in difference:
      G.remove_node(node)

    return G

############################################################################################
#                                       QUERIES                                            #
############################################################################################

def get_response(prompt, client, provider, model):
    '''
    Function to get the LLM response for a prompt
    :param prompt: instruction prompt
    :param client: OpenAI client
    :param provider: openai, llmstudio or mistral
    :param model: OpenAI model (default gpt-3.5-turbo-1106)
    '''

    if provider == 'openai' or provider=='llmstudio':

        if provider == 'llmstudio':
            prompt = '[INST] '+prompt + '[/INST]'
        messages = [{"role": "user", "content": prompt}]
        
        conversation = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        seed = 1998)

    elif provider == 'mistral':
        messages = [ChatMessage(role="user", content=prompt)]

        # No streaming
        conversation = client.chat(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0)


    response = conversation.choices[0].message.content

    # Consistency changes for the mistal7B outputs
    if response.startswith('[') and response.endswith(']'):
        response = response[1:-1]
    response = response.replace('```json','').replace('```','')
    
    return response

def case_1_query(elements, text, mention):
    '''
    Write the query for the first case (i.e., all the elements are subclasses) 
    :param elements: all the candidate subclasses
    :param text: text in which the mention is found
    :param mention: mention of the entity in the text
    '''

    new_elements = elements + ['None'] 

    prompt = 'This is an Entity Disambiguation task. \nGiven the mention between [START_ENT] and [END_ENT]\n'+text
    prompt = prompt +'\n\nTo which of these categories in the list the mention '+ mention +' refers to?\n'

    for candidate in new_elements:
        prompt = prompt +candidate+'\n'
    prompt += '\nThe name must be in the list and in json format: {"category": "categoryName"}'

    return prompt


def case_2_query(elements, text, mention):
    '''
    Write the query for the first case (i.e., all the elements are entities) 
    :param elements: all the candidate entities (without the strange character changes)
    :param text: text in which the mention is found
    :param mention: mention of the entity in the text
    '''
    wiki = wikipediaapi.Wikipedia('MyProjectName', 'en')

    prompt = 'This is an Entity Disambiguation task. \nGiven the mention between [START_ENT] and [END_ENT]\n'+text
    prompt = prompt +'\n\nGiven this context:\n'

    for candidate in elements:
        page_py = wiki.page(candidate)
        page_py.exists()
        summary = page_py.summary[0:250]
        prompt = prompt + candidate+': '+summary+'\n'
    prompt = prompt + 'None\n'
    prompt += '\nTo which of these entities in the list the mention '+ mention +' refers to ['+', '.join(elements) +', None]? Answer in json format: {"entity": "entityName"}'

    return prompt

def case_3_query(elements, text, mention):
    '''
    Write the query for the third case (i.e., elements are subclasses and entities) 
    :param elements: all the candidate subclasses
    :param text: text in which the mention is found
    :param mention: mention of the entity in the text
    '''

    prompt = 'This is an Entity Disambiguation task. \nGiven the mention between [START_ENT] and [END_ENT]\n'+text
    prompt = prompt +'\n\nTo which of these categories in the list the mention '+ mention +' refers to?\n'

    for candidate in elements:
        prompt = prompt +candidate+'\n'
    prompt = prompt +'Other'+'\n'
    prompt += '\nThe name must be in the list, and in json format: {"category": "categoryName"}'

    return prompt

def case_none_query(elements, text, mention):
    '''
    Write the query the None case, in which we assess the whole initial candidate list
    :param elements: all the candidate entities (without the strange character changes)
    :param text: text in which the mention is found
    :param mention: mention of the entity in the text
    '''

    prompt = 'This is an Entity Disambiguation task. \nGiven the mention between [START_ENT] and [END_ENT]\n'+text
    prompt = prompt +'\n\nTo which of these entities the mention '+ mention +' refers to? \n'
    prompt = prompt + '['+', '.join(elements) +']'
    
    prompt += '\nThe entity must be in the list, answer in json format: {"entity": "entityName"}'

    return prompt



def case_check_query(response, text, mention):
    '''
    Write the query for the check when the answer is given by discarding
    :param elements: all the candidate subclasses
    :param text: text in which the mention is found
    :param mention: mention of the entity in the text
    '''

    prompt = 'This is an Entity Disambiguation task. \nGiven the entity between [START_ENT] and [END_ENT]\n'+text
    prompt = prompt +'\n\nDoes the mention '+ mention +' refer to \''+ response +'\'?'

    prompt += '\nAnswer only Yes or No in json format: {"answer": "Yes|No"}'

    return prompt

def case_exact_query(candidates, bad_llm):
    '''
    Create prompt for asking the llm to give a correct input
    :param candidates: list of the candidates the llm should have answered
    :param bad_llm: previously provided answer
    '''
    prompt = 'Given this mention ['+ bad_llm +'] choose the name from the list that matches it:\n'
    for candidate in candidates:
        prompt = prompt +'-'+candidate+'\n'
    prompt = prompt + '\nThe name must be in the list, in json format: {"entity": "entityName"}'
    return prompt


def check_errors(candidates, full_candidates ,text, mention, prompt,client,llm_provider, model):
    '''
    Check errors when the LLM produced a bad result or the class was potentially incorrect
    '''

    if candidates[0] != 'BadLLM':
        prompt = case_check_query(candidates[0],text,mention)
        llm = json.loads(get_response(prompt,client, provider=llm_provider, model=model))['answer']
    else:
        llm = 'No'

    if llm == 'No':
        prompt = case_none_query(full_candidates,text,mention)
        llm = json.loads(get_response(prompt, client, provider=llm_provider, model=model))['entity']
        candidates = [llm]

    return candidates

############################################################################################
#                              CLEAN URI AND ANSWERS                                       #
############################################################################################

def yago_input(elements, ns):
    '''
    Function to transform elements into YAGO url
    :param elements: list of elements to be transformed
    :param ns: if yago namespace must be included or not in the string (True, False)
    '''
    elements = [i.replace(' ','_').replace('(','_u0028_').replace(')','_u0029_').replace('.','_u002E_').replace('\'','_u0027_').replace('&','_u0026_').replace(',','_u002C_') for i in elements]
    
    if ns:
        elements = ['http://yago-knowledge.org/resource/'+i for i in elements]

    return elements


def wikipedia_input(elements):
    '''
    Given candidates in YAGO syntax, transform it back to wikipedia
    :param elements: list of elements to be transformed
    '''

    elements = [i.replace('_u0028_','(').replace('_u0029_',')').replace('_u002E_','.').replace('_u0027_''\'','\'').replace('_u0026_','&').replace('_u002C_',',') for i in elements]
    
    return elements

def clean(elements):
    '''
    Given candidates in YAGO syntax, transform it back to original answer
    :param elements: list of elements to be transformed
    '''

    elements = [i.replace('_u0028_','(').replace('_u0029_',')').replace('_u002E_','.').replace('_u0027_''\'','\'').replace('_u0026_','&').replace('_u002C_',',').replace('_',' ') for i in elements]
    
    return elements

def exact_match(response,candidates,client,llm_provider, model,similarity=False):
    '''
    Given an LLM answer that does not exactly match the expected responses, an answer is given
    :param response: output of the LLM
    :param candidates: possible candidates to match
    :param config: in which case of the algorithm are we
    :param similarity: if True, a similarity check is performed between the words
    '''
    if similarity:
        answer = min(candidates, key=lambda word: Levenshtein.distance(response, word))
    else:
        prompt = case_exact_query(candidates,response)
        answer = json.loads(get_response(prompt,client, provider=llm_provider, model=model))['entity']

    return answer