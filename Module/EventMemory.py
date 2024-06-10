"""
Contains the long-short memory modules of LD-Agent
"""
import time
import math
import spacy
import chromadb
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction, DefaultEmbeddingFunction


class EventMemory():
    def __init__(self, client, sample_id, logger, args, memory_cache=None):
        self.args = args
        self.logger = logger
        self.embedding_function = DefaultEmbeddingFunction()
        self.data_loader = ImageLoader()
        self.lemma_tokenizer = spacy.load("en_core_web_sm")
        dbclient_init = False

        sleep_time = 1
        while dbclient_init == False:
            try:
                if memory_cache:
                    self.dbclient = chromadb.PersistentClient(path=memory_cache)
                    dbclient_init = True

                else:
                    # self.dbclient = chromadb.Client(tenant=f"tenant_{sample_id}", database=f"database_{sample_id}")
                    self.dbclient = chromadb.Client()
                    dbclient_init = True
                    self.logger.info(f"{sample_id} dbclient init completed!")
            except:
                time.sleep(sleep_time)
                sleep_time += 1

        self.LLMclient = client

        self.usr_name = args.usr_name
        self.agent_name = args.agent_name

        self.current_time_pass = 0

        self.overall_retrieve_score = 0.0
        self.overall_retrieve_count = 0

        self.collection = self.dbclient.create_collection(
            name=f"collection_{sample_id}", 
            embedding_function=self.embedding_function, 
            data_loader=self.data_loader,
            get_or_create=(memory_cache != None))
        
        self.short_term_memory = []
        

    def store(self, ids, key, metadata, datatype='text'):
        if datatype not in ['image', 'text']:
            raise ValueError("input_type must be 'image' or 'text'")
        
        if type(key) != list:
            key = [key]

        if type(ids) != list:
            ids = [str(ids)]
        
        if datatype == 'image':
            self.collection.add(
                ids=ids,
                images=key, # A list of numpy arrays representing images
                metadatas=metadata
            )

        if datatype == 'text':
            self.collection.add(
                ids=ids,
                documents=key, # A list of strings representing texts
                metadatas=metadata
            )


    def relevance_retrieve(self, ori_query, n_results=10, dist_thres=0.5, filter=None, current_time=0.0, datatype='text', decay_temp=1E-7):
        if datatype not in ['image', 'text']:
            raise ValueError("input_type must be 'image' or 'text'")
        
        if type(ori_query) != list:
            ori_query = [ori_query]

        query = []
        for idx, query_item in enumerate(ori_query):
            # tokenized keys
            tokenized_item = self.lemma_tokenizer(query_item)

            query_nouns_item = list(set([token.lemma_ for token in tokenized_item if token.pos_ == "NOUN"]))
            merged_nouns_str = ",".join(query_nouns_item)
            query.append(merged_nouns_str)
        
        if self.args.ori_mem_query:
            query = ori_query

        if datatype == 'image':
            results=self.collection.query(
                query_images=query, # A list of numpy arrays representing images
                n_results = n_results,
                where=filter,
            )

        if datatype == 'text':
            results=self.collection.query(
                query_texts=query, # A list of strings representing texts
                n_results = n_results,
                where=filter,
            )


        # sorted by time
        metadata_list = []
        best_memory = {'idx': 0, 'overall_score': 0.0, 'overlap_score': 0.0, 'overlap_count': 0, 'query_nouns_item': 0, 'retrieved_nouns_item': 0, 'distance': dist_thres}
        empty_flag = False
        if self.collection.count() > 0:
            # compute overlap
            for idx, retrieved_item in enumerate(results['metadatas'][0]):
                distance = results['distances'][0][idx]
                # filter too large distance
                # if distance >= dist_thres:
                #     continue

                # compute overlap topics number
                retrieved_nouns_item = retrieved_item['topics'].split(',')
                overlap_count = len(set(query_nouns_item) & set(retrieved_nouns_item))

                if (len(query_nouns_item) == 0) or (len(retrieved_nouns_item) == 0):
                    overlap_score = 0
                else:
                    overlap_score = 0.5 * (overlap_count / len(query_nouns_item)) + 0.5 * (overlap_count / len(retrieved_nouns_item))

                # compute time decay weight
                time_gap = current_time - retrieved_item['time']
                time_decay_coe = math.exp(-decay_temp * time_gap)

                overall_score = time_decay_coe * overlap_score

                if self.args.ori_mem_query and (results['distances'][0][idx] < dist_thres) and (results['distances'][0][idx] < best_memory['distance']):
                    empty_flag = True
                    best_memory['idx'] = idx
                    best_memory['overall_score'] = overall_score
                    best_memory['overlap_score'] = overlap_score
                    best_memory['overlap_count'] = overlap_count
                    best_memory['query_nouns_item'] = query_nouns_item
                    best_memory['retrieved_nouns_item'] = retrieved_nouns_item
                    best_memory['distance'] = results['distances'][0][idx]

                elif (overlap_count > 0) and (overall_score >= best_memory['overall_score']):
                    empty_flag = True
                    best_memory['idx'] = idx
                    best_memory['overall_score'] = overall_score
                    best_memory['overlap_score'] = overlap_score
                    best_memory['overlap_count'] = overlap_count
                    best_memory['query_nouns_item'] = query_nouns_item
                    best_memory['retrieved_nouns_item'] = retrieved_nouns_item      
                    best_memory['distance'] = results['distances'][0][idx]              

            if empty_flag:
                metadata_list.append(results['metadatas'][0][best_memory['idx']])

                self.overall_retrieve_score += best_memory['overlap_score']
                self.overall_retrieve_count += 1
                # self.logger.info(f'Average Retrieval Scores: {self.overall_retrieve_score / self.overall_retrieve_count}')


        return metadata_list

    def direct_retrieve(self, ori_query, n_results=10, dist_thres=0.5, filter=None, current_time=0.0, datatype='text', decay_temp=1E-7):
        if datatype not in ['image', 'text']:
            raise ValueError("input_type must be 'image' or 'text'")
        
        if type(ori_query) != list:
            ori_query = [ori_query]

        query = ori_query

        if datatype == 'image':
            results=self.collection.query(
                query_images=query, # A list of numpy arrays representing images
                n_results = n_results,
                where=filter,
            )

        if datatype == 'text':
            results=self.collection.query(
                query_texts=query, # A list of strings representing texts
                n_results = n_results,
                where=filter,
            )

        # results=self.collection.query(query_texts=['hobby,grill'], n_results = n_results,where=filter)

        latest_filter = {
                        "idx": {
                            "$gte": self.collection.count() - 1},
        }

        # sorted by time
        metadata_list = [results['metadatas'][0][idx] for idx, distance in enumerate(results['distances'][0]) if distance < dist_thres]

        return metadata_list



    def context_retrieve(self, query, n_results=10, current_time=0, datatype='text'):
        one_hour_seconds = 60 * 60

        # change session
        if (len(self.short_term_memory) > 0) and (current_time - self.short_term_memory[-1]['time']) > one_hour_seconds:
            last_session_context = [f"(line {context_ids + 1}) " + f"{context_memory['dialog']}." for context_ids, context_memory in enumerate(self.short_term_memory)]
            merged_last_session_context = "\n".join(last_session_context)
            last_session_summary = self.context_summarize(merged_last_session_context, len(last_session_context))
            # self.logger.info(last_session_summary)

            # tokenized keys
            tokenized_item = self.lemma_tokenizer(merged_last_session_context)

            context_nouns_item = list(set([token.lemma_ for token in tokenized_item if token.pos_ == "NOUN"]))
            merged_nouns_str = ",".join(context_nouns_item)

            # store last session's summary
            self.store(self.collection.count(), merged_nouns_str, MetaData(idx = self.collection.count(), dialog="", time = self.short_term_memory[-1]['time'], topics=merged_nouns_str, datatype='text', summary=last_session_summary).to_dict(), datatype='text')

            # initilize current session
            self.short_term_memory = []
            data = {"idx": len(self.short_term_memory), "time": current_time, "dialog": f"{self.usr_name}: {query}"}
            self.short_term_memory.append(data)

        else:
            # store user query in short-term memory
            data = {"idx": len(self.short_term_memory), "time": current_time, "dialog": f"{self.usr_name}: {query}"}
            self.short_term_memory.append(data)

        if len(self.short_term_memory) >= n_results:
            sorted_metadatas = self.short_term_memory[-n_results:]

        else:
            sorted_metadatas = self.short_term_memory

        return sorted_metadatas


    def retrieve(self, query, n_results=10, filter=None, datatype='text'):
        if datatype not in ['image', 'text']:
            raise ValueError("input_type must be 'image' or 'text'")
        
        if type(query) != list:
            query = [query]
        
        if datatype == 'image':
            results=self.collection.query(
                query_images=query, # A list of numpy arrays representing images
                n_results = n_results,
                where=filter,
            )

        if datatype == 'text':
            results=self.collection.query(
                query_texts=query, # A list of strings representing texts
                n_results = n_results,
                where=filter,
            )

        return results
    
    
    
    def context_summarize(self, context, length):
        """
        Summarize the context.
        """

        sys_prompt = f"""You are good at extracting events and summarizing them in brief sentences. You will be shown a conversation between {self.usr_name} and {self.agent_name}.\n"""

        user_prompt = f"""#Conversation#:\n{context}.\n""" \
                    + f"""Based on the Conversation, please summarize the main points of the conversation with brief sentences in English, within 20 words.\nSUMMARY:"""  

        summary = self.LLMclient.employ(sys_prompt, user_prompt, "EventSummary")

        return summary
    

class MetaData():
    def __init__(self, keys=['idx', 'dialog', 'time', 'datatype', 'summary', 'topics'], idx=0, dialog='', time=time.time(), topics='', datatype=None, summary=''):
        self.idx=idx
        self.dialog = dialog
        self.time = time
        self.datatype=datatype # 'None', 'image', 'text'
        self.summary=summary
        self.keys=keys
        self.topics = topics

        self.value_verification()

    def to_dict(self):
        return {key: getattr(self, key) for key in self.keys}
    
    def value_verification(self):
        if self.datatype not in ['image', 'text']:
            raise ValueError("datatype must be 'image' or 'text'")
        
        