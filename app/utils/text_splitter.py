from typing import List

class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """
        Recursively splits text into chunks.
        """
        final_chunks = []
        if self._length_function(text) <= self.chunk_size:
            return [text]

        # Use the first separator that produces valid splits
        for separator in self.separators:
            if separator == "":
                # Fallback to character splitting
                return self._split_by_chars(text)
            
            if separator in text:
                splits = text.split(separator)
                chunks = self._merge_splits(splits, separator)
                if chunks:
                    final_chunks = chunks
                    break
        
        return final_chunks

    def _length_function(self, text: str) -> int:
        return len(text)

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        docs = []
        current_doc = []
        total = 0
        for d in splits:
            _len = self._length_function(d)
            if total + _len + (len(separator) if current_doc else 0) > self.chunk_size:
                if total > self.chunk_size:
                    # If a single split is larger than chunk_size, we might need to recurse on it
                    pass # Simplified: just keeping it or we could recurse. 
                
                if current_doc:
                    doc = separator.join(current_doc)
                    if doc:
                        docs.append(doc)
                
                # Handling overlap (simplified)
                while total > self.chunk_overlap or (total + _len > self.chunk_size and total > 0):
                     total -= self._length_function(current_doc[0]) + len(separator)
                     current_doc.pop(0)

                current_doc = []
                total = 0
            
            current_doc.append(d)
            total += _len + (len(separator) if len(current_doc) > 1 else 0)
        
        doc = separator.join(current_doc)
        if doc:
            docs.append(doc)
            
        return docs

    def _split_by_chars(self, text: str) -> List[str]:
        # Fallback simplistic character splitter
        return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size-self.chunk_overlap)]
