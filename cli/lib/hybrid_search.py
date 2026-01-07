import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import load_movies


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query, k=60, limit=5):
        # 1. Fetch expanded result sets (500x limit)
        fetch_limit = limit * 500
        bm25_results = self._bm25_search(query, fetch_limit)
        semantic_results = self.semantic_search.search_chunks(query, fetch_limit)

        combined_data = {}

        # 2. Track BM25 Ranks
        for rank, res in enumerate(bm25_results, 1):
            doc_id = res['id']
            combined_data[doc_id] = {
                'bm25_rank': rank,
                'semantic_rank': None,
                'doc': res,
                'rrf_score': rrf_score(rank, k)
            }

        # 3. Track Semantic Ranks and Combine Scores
        for rank, res in enumerate(semantic_results, 1):
            doc_id = res['id']
            score = rrf_score(rank, k)
            if doc_id in combined_data:
                combined_data[doc_id]['semantic_rank'] = rank
                combined_data[doc_id]['rrf_score'] += score
            else:
                combined_data[doc_id] = {
                    'bm25_rank': None,
                    'semantic_rank': rank,
                    'doc': res,
                    'rrf_score': score
                }

        # 4. Format and Sort results
        final_results = []
        for doc_id, data in combined_data.items():
            final_results.append({
                "title": data['doc']['title'],
                "description": data['doc']['document'],
                "rrf_score": data['rrf_score'],
                "bm25_rank": data['bm25_rank'],
                "semantic_rank": data['semantic_rank']
            })

        # Sort by RRF score descending
        final_results.sort(key=lambda x: x['rrf_score'], reverse=True)
        return final_results[:limit]
    
    def weighted_search(self, query, alpha, limit):
        fetch_limit = limit * 500
        bm25_results = self._bm25_search(query, fetch_limit)
        semantic_results = self.semantic_search.search_chunks(query, fetch_limit)

        bm25_raw_scores = [r["score"] for r in bm25_results]
        semantic_raw_scores = [r["score"] for r in semantic_results]

        norm_bm25 = normalize(bm25_raw_scores)
        norm_semantic = normalize(semantic_raw_scores)

        combined_data = {}

        for i, res in enumerate(bm25_results):
            doc_id = res['id']
            combined_data[doc_id] = {
                'keyword': norm_bm25[i],
                'semantic': 0.0,  # Default if not found in semantic results
                'doc': res
            }
        
        for i, res in enumerate(semantic_results):
            doc_id = res['id']
            if doc_id in combined_data:
                combined_data[doc_id]['semantic'] = norm_semantic[i]
            else:
                combined_data[doc_id] = {
                    'keyword': 0.0,
                    'semantic': norm_semantic[i],
                    'doc': res
                }

        hybrid_results = []
        for doc_id, data in combined_data.items():
            # Apply alpha weighting
            h_score = hybrid_score(data['keyword'], data["semantic"], alpha)

            result = data['doc']
            result['score'] = round(h_score, 4)
            # Standardizing result structure
            hybrid_results.append(result)


        # 5. Calculate hybrid scores
        final_results = []
        for doc_id, data in combined_data.items():
            # Formula: (alpha * semantic) + ((1 - alpha) * bm25)
            h_score = hybrid_score(data['keyword'], data["semantic"], alpha)
            
            # Prepare data for the CLI formatter
            final_results.append({
                "title": data['doc']['title'],
                "description": data['doc']['document'],
                "hybrid_score": h_score,
                "bm25_score": data['keyword'],
                "semantic_score": data['semantic']
            })

        # 6. Sort by hybrid score in descending order
        final_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return final_results[:limit]
        

def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score

def normalize(scores):
    if not scores:
        return []
    min_s, max_s = min(scores), max(scores)
    if min_s == max_s:
        return [1.0 for _ in scores]
    return [(s - min_s) / (max_s - min_s) for s in scores]

def hybrid_search(query, alpha, limit):
    searcher = HybridSearch(load_movies())
    # Initialize both indices as needed
    results = searcher.weighted_search(
        query, 
        alpha, 
        limit
    )
    
    return results

def rrf_score(rank, k=60):
    """Calculate the RRF score for a single rank."""
    return 1 / (k + rank)

def rrf_search_helper(query, k, limit):
    searcher = HybridSearch(load_movies())
    return searcher.rrf_search(query, k, limit)