"""
Filtro temporale sui gesti: trasforma le predizioni frame per frame (instabili nelle
transizioni) in un gesto stabile oppure None ("nessun comando").
"""

from collections import Counter, deque


class GestureFilter:
    def __init__(self, window=7, min_agreement=0.8, min_confidence=0.7):
        self.recent = deque(maxlen=window)    # ultime predizioni: le più vecchie escono da sole
        self.min_agreement = min_agreement    # quota minima di frame che devono concordare
        self.min_confidence = min_confidence  # sotto questa confidenza il frame vale "nessun gesto"

    def update(self, gesture, confidence):
        """Riceve la predizione di un frame, restituisce il gesto stabile oppure None."""
        if confidence < self.min_confidence:
            self.recent.append(None)
        else: 
            self.recent.append(gesture)
       
        if len(self.recent) < self.recent.maxlen:
            return None
        
        value, count = Counter(self.recent).most_common(1)[0]
        share = count / len(self.recent)
        if share >= self.min_agreement and value is not None:
            return value 
        return None