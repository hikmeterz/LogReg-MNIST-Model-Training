from typing import List
import random
import math

class LogisticRegression:
    def __init__(self, learning_rate: float, epoch: int, batch_size: int):
        """
        Logistic regression modelinin başlatıcısı (constructor) fonksiyonu.

        Parametreler:
        - learning_rate (float): Öğrenme oranı, modelin ağırlık güncellemelerinin hızını belirler.
        - epoch (int): Eğitim sırasında modelin kaç kez tüm veri setini göreceğini belirten epoch sayısı.
        - batch_size (int): Her eğitim adımında kullanılacak örnek sayısı, batch boyutu.
        """
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.weights = None
        self.biases = None

    def create_mini_batches(self, X, y):
        """
        Eğitim verilerini küçük gruplara (mini-batches) bölen fonksiyon.
        
        Parametreler:
        - X: Özellik (feature) matrisi.
        - y: Hedef (label) vektörü.
        
        Geri Dönen:
        - mini_batches: Özellik ve hedeflerin küçük gruplar halinde tutulduğu liste.
        """
        m = len(X) # Toplam örnek sayısını alır.
        indices = list(range(m)) # 0'dan m-1'e kadar olan indeksleri içeren bir liste oluşturur.
        random.shuffle(indices) # İndeks listesini karıştırarak veri setini rastgele hale getirir.
        
        mini_batches = []
        for k in range(0, m, self.batch_size): # Her batch için seçilecek indeksleri belirler ve X ile y'yi buna göre böler.
            X_mini = [X[i] for i in indices[k:k+self.batch_size]]
            y_mini = [y[i] for i in indices[k:k+self.batch_size]]
            mini_batches.append((X_mini, y_mini))
            
        return mini_batches # Mini-batch'lerin listesini döner.

    def softmax(self, z):
        """
        Softmax fonksiyonu, giriş matrisindeki her elemanın olasılık dağılımını hesaplar.
        
        Parametreler:
        - z: Giriş matrisi (genellikle logits veya ham tahminler).
        
        Geri Dönen:
        - softmax_z: Giriş matrisindeki her elemanın softmax olasılıkları.
        """
        # Her elemanın e^i değerini hesaplar ve sayıların taşmasını önlemek için
        # her satırdan maksimum değeri çıkarır.
       
        exp_z = [[math.exp(i - max(row)) for i in row] for row in z]
        softmax_z = [[i / sum(row) for i in row] for row in exp_z]
        return softmax_z  # Softmax olasılık matrisini döner.

    def forward(self, X):
        """
        İleri besleme fonksiyonu, modelin giriş verileri üzerinden tahminler yapmasını sağlar.
        
        Parametreler:
        - X: Özellik (feature) matrisi.
        
        Geri Dönen:
        - softmax_z: Giriş verilerinin softmax olasılıkları.
        """
        # Z matrisini hesaplar. Z, giriş verileri (X) ile ağırlıklar (weights) matrisinin çarpımı
        # ve bias terimlerinin eklenmesiyle elde edilir.
        z = [[sum(x_i * w_i for x_i, w_i in zip(x_row, w_col)) + self.biases[0][j] for j, w_col in enumerate(zip(*self.weights))] for x_row in X]
        # Softmax fonksiyonunu kullanarak Z matrisini softmax olasılıklarına dönüştürür.
        return self.softmax(z)

    def calculate_gradient_W(self, X_mini, y_pred, y_mini):
        """
        Ağırlık matrisinin gradyanını (türevini) hesaplayan fonksiyon.
        
        Parametreler:
        - X_mini: Mini-batch içindeki özellik (feature) matrisi.
        - y_pred: Modelin mini-batch için tahmin ettiği değerler.
        - y_mini: Mini-batch içindeki gerçek hedef (label) değerleri.
        
        Geri Dönen:
        - gradient_W: Ağırlık matrisinin gradyanı (türevi).
        """
        # X_mini matrisinin transpozunu (transpose) alır.
        X_mini_T = list(map(list, zip(*X_mini)))
        # Tahmin edilen değerler ile gerçek değerler arasındaki farkı hesaplar.
        y_diff = [[yp - ym for yp, ym in zip(y_pred_row, y_mini_row)] for y_pred_row, y_mini_row in zip(y_pred, y_mini)]
        # Ağırlık matrisinin gradyanını hesaplar. Bu, X_mini_T'nin her sütunu ile y_diff'ün
        # her sütununun çarpımlarının toplamının batch boyutuna bölünmesiyle elde edilir.
        
        gradient_W = [[sum(x * y for x, y in zip(X_col, y_diff_col)) / self.batch_size for y_diff_col in zip(*y_diff)] for X_col in X_mini_T]
        return gradient_W # Hesaplanan gradyan (türev) matrisini döner.

    def calculate_gradient_b(self, y_pred, y_mini):
        """
        Bias terimlerinin gradyanını (türevini) hesaplayan fonksiyon.
        
        Parametreler:
        - y_pred: Modelin mini-batch için tahmin ettiği değerler.
        - y_mini: Mini-batch içindeki gerçek hedef (label) değerleri.
        
        Geri Dönen:
        - gradient_b: Bias terimlerinin gradyanı (türevi).
        """
        # Tahmin edilen değerler ile gerçek değerler arasındaki farkı hesaplar.
        y_diff = [[yp - ym for yp, ym in zip(y_pred_row, y_mini_row)] for y_pred_row, y_mini_row in zip(y_pred, y_mini)]
        # Bias terimlerinin gradyanını hesaplar. Bu, y_diff'ün her sütununun
        # toplamının, o sütunun eleman sayısına bölünmesiyle elde edilir.
        
        gradient_b = [sum(col) / len(col) for col in zip(*y_diff)]
        return [gradient_b]

    def loss(self, y_true, y_pred):
        """
        Log-loss (çapraz entropi kaybı) fonksiyonunu hesaplayan fonksiyon.
        
        Parametreler:
        - y_true: Gerçek hedef (label) değerleri.
        - y_pred: Modelin tahmin ettiği değerler.
        
        Geri Dönen:
        - mean_loss: Ortalama log-loss değeri.
        """
        
        
        epsilon = 1e-8 # Logaritmanın içindeki sıfır hatalarını önlemek için küçük bir sabit.
        # Her örnek için log-loss değerlerini hesaplar.
        loss_values = [sum(y_t * math.log(y_p + epsilon) for y_t, y_p in zip(y_true_row, y_pred_row)) for y_true_row, y_pred_row in zip(y_true, y_pred)]
         # Log-loss değerlerinin ortalamasını alarak mean_loss'u hesaplar.
        
        mean_loss = -sum(loss_values) / len(loss_values)
        return mean_loss

    def fit(self, X: List[List[int]], y: List[List[int]]):
        """
        Modeli eğitim verileri üzerinde eğiten fonksiyon.
        
        Parametreler:
        - X: Eğitim veri setinin özellik (feature) matrisi.
        - y: Eğitim veri setinin hedef (label) matrisi.
        
        Geri Dönen:
        - loss_history: Her epoch'taki kayıp değerlerini içeren liste.
        """
        
        m = len(X)
        n = len(X[0]) if X else 0
        k = len(y[0]) if y and y[0] else 0
        # Ağırlık matrisini ve bias vektörünü sıfırlarla başlatır.
        
        self.weights = [[0.0 for _ in range(k)] for _ in range(n)]
        self.biases = [[0.0 for _ in range(k)]]
        loss_history = []

        for epoch in range(self.epoch):
            mini_batches = self.create_mini_batches(X, y)
            for X_mini, y_mini in mini_batches:
                y_pred = self.forward(X_mini)
                gradient_W = self.calculate_gradient_W(X_mini, y_pred, y_mini)
                gradient_b = self.calculate_gradient_b(y_pred, y_mini)
                # Ağırlıkları ve biasları gradyan adımıyla günceller.
                
                self.weights = [[w - self.learning_rate * g for w, g in zip(weight_row, grad_row)] for weight_row, grad_row in zip(self.weights, gradient_W)]
                self.biases = [[b - self.learning_rate * g for b, g in zip(bias_row, grad_row)] for bias_row, grad_row in zip(self.biases, gradient_b)]

            y_pred_full = self.forward(X)
            l = self.loss(y, y_pred_full)
            loss_history.append(l)
            print(f"Epoch {epoch}, Loss: {l}")

        return loss_history

    def predict(self, X: List[List[int]]):
        y_pred = self.forward(X)
        return [max(range(len(row)), key=row.__getitem__) for row in y_pred]
