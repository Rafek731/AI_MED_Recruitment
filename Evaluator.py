class Evaluator:
    def __init__(self):
        self.TP = 0
        self.TN = 0 
        self.FP = 0
        self.FN = 0
        self.idx= 0
        self.print_head()
    

    @property
    def accuracy(self) -> float:
        try:
            return (self.TN+self.TP)/(self.TN+self.TP+self.FP+self.FN)
        except ZeroDivisionError:
            return 0.
        

    @property
    def precision(self) -> float:
        try:
            return self.TP/(self.TP+self.FP)
        except ZeroDivisionError:
            return 0.
        
    
    @property
    def recall(self) -> float:
        try:
            return self.TP/(self.TP+self.FN)
        except ZeroDivisionError:
            return 0.
        
        
    @property
    def f1_score(self) -> float:
        try:
            return 2*self.recall*self.precision/(self.precision + self.recall)
        except ZeroDivisionError:
            return 0.
    

    def print_head(self) -> None:
        print('Evaluation metrics:\n num  | Accuracy | Precision | Recall | F1_Score |')


    def print(self) -> tuple[tuple[str, float]]:
        print(f'{self.idx:^6}|{self.accuracy:^10.2f}|{self.precision:^11.2f}|{self.recall:^8.2f}|{self.f1_score:^10.2f}|')
        return (('Accuracy', self.accuracy),('Precision', self.precision), ('Recall', self.recall), ('F1 Score', self.f1_score))

    
    def print_final(self) -> tuple[float]:
        print(f'\nFinal: \naccuracy={self.accuracy:.2f} | precision={self.precision:.2f} | recall={self.recall:.2f} | f1_score={self.f1_score:.2f}')
        return (('Accuracy', self.accuracy),('Precision', self.precision), ('Recall', self.recall), ('F1 Score', self.f1_score))

    
    def judge(self, prediction: int, label: int) -> None:
        self.idx += 1
        if prediction == 0 and label == 0:
            self.TN += 1
        elif prediction == 1 and label == 1:
            self.TP += 1
        elif prediction == 0 and label == 1:
            self.FN += 1
        else:
            self.FP += 1