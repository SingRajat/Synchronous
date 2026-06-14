from langchain_core.runnables import RunnableLambda
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from app.llm.generator import Get_LLM
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage


def Retriever(db,bm25_retriever):
    dense_retriever=db.as_retriever(
        search_kwargs={"k": 3}
    )

    reranker=FlashrankRerank(
        top_n=3
    )

    llm=Get_LLM()

    def get_unique_docs(queries):
        doc_scores={}
        for q in queries:
            vector_docs=dense_retriever.invoke(q)
            bm25_docs=bm25_retriever.invoke(q)

            for rank,doc in enumerate(bm25_docs):
                doc_id = doc.metadata.get("id", doc.page_content)
                if doc_id not in doc_scores:
                    doc_scores[doc_id]={"docs":doc,"score":0}
                doc_scores[doc_id]["score"]+=1/(rank+60)
            
            for rank,doc in enumerate(vector_docs):
                doc_id = doc.metadata.get("id", doc.page_content)
                if doc_id not in doc_scores:
                    doc_scores[doc_id]={"docs":doc,"score":0}
                doc_scores[doc_id]["score"]+=1/(rank+60)

        # 2. Sort outside the query loop, so scores accumulate across all queries
        ranked=sorted(doc_scores.values(),key=lambda x:x["score"], reverse=True )
        top_docs=[item["docs"] for item in ranked][:5]
        
        if not top_docs:
            return []
            
        RR_Comp_docs=reranker.compress_documents(top_docs,query=queries[0])

        final_compressed_docs=[]

        Compress_prompt=SystemMessage(content="""Task: You are an expert at document compression for RAG. Your job is to remove irrelevant sentences from documents while preserving the semantic meaning relevant to the query. 

        Input Format:
        [DOCUMENT] = The text of a document after initial retrieval.
        [QUERY] = The user's question.
        
        Instructions:
        1. Read [DOCUMENT] and [QUERY] carefully. 
        2. Extract only the sentences or phrases from [DOCUMENT] that are necessary to answer the [QUERY].
        3. Remove all other content (irrelevant details, redundant sentences, filler text).
        4. Maintain the original meaning and key facts.
        5. If nothing in the document is relevant, return exactly the string "NO_MATCH".
        6. Do NOT output any text other than the compressed document.
        
        Example 1:
        [DOCUMENT] = "The Eiffel Tower is in Paris, France. It was completed in 1889. It is 330 meters tall. Many tourists visit it."
        [QUERY] = "Where is the Eiffel Tower?"
        Output: "The Eiffel Tower is in Paris, France."

        Example 2:
        [DOCUMENT] = "The meeting is on Tuesday. Please bring the Q3 report. The agenda includes budget review. Lunch will be provided."
        [QUERY] = "When is the meeting?"
        Output: "The meeting is on Tuesday."

        Example 3:
        [DOCUMENT] = "The car is red. It has four doors. The engine size is 2.0L."
        [QUERY] = "What color is the sky?"
        Output: "NO_MATCH"
        
        Now, compress the following document:
        [DOCUMENT] = {document_text}
        [QUERY] = {query}
        """)
        for doc in RR_Comp_docs:
            user_prompt=HumanMessage(content=f"[QUERY]:{queries[0]}\n[DOCUMENT]:{doc.page_content}")
            response=llm.invoke([Compress_prompt,user_prompt])
            extracted_text=response.content.strip()

            if extracted_text!="NO_MATCH":
                final_compressed_docs.append(Document(page_content=extracted_text,
                metadata=doc.metadata))

        return final_compressed_docs  
    return RunnableLambda(get_unique_docs)
 