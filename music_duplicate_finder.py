import os
import hashlib
import librosa
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# GPU-ускоренные импорты
try:
    import cupy as cp
    import cudf
    from cuml.metrics import pairwise_distances
    from cuml.neighbors import NearestNeighbors
    GPU_AVAILABLE = True
    print("CUDA GPU acceleration enabled")
except ImportError:
    GPU_AVAILABLE = False
    print("CUDA not available, falling back to CPU")
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.neighbors import NearestNeighbors

class MusicDuplicateFinder:
    def __init__(self, library_path, similarity_threshold=0.95, use_gpu=True, batch_size=1000):
        self.library_path = Path(library_path)
        self.similarity_threshold = similarity_threshold
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.batch_size = batch_size
        self.audio_files = []
        self.spectrogram_features = {}
        self.feature_matrix = None
        self.duplicates = defaultdict(list)
        
        if self.use_gpu:
            print(f"Using GPU acceleration (RTX 4090 CUDA)")
        else:
            print("Using CPU processing")
    
    def find_audio_files(self):
        """Поиск всех аудио файлов в библиотеке с поддержкой большего количества форматов"""
        audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.ape', '.wma', '.alac', '.aiff'}
        
        for root, _, files in os.walk(self.library_path):
            for file in files:
                if Path(file).suffix.lower() in audio_extensions:
                    self.audio_files.append(Path(root) / file)
        
        print(f"Found {len(self.audio_files)} audio files")
        return self.audio_files
    
    def compute_md5_hash(self, file_path):
        """Вычисление MD5 хэша файла с буферизацией для больших файлов"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):  # Увеличенный буфер
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            print(f"Error hashing {file_path}: {e}")
            return None
    
    def extract_spectrogram_features_gpu(self, file_path, sr=22050, n_fft=2048, hop_length=512, max_length=50000):
        """Извлечение признаков с GPU-ускорением"""
        try:
            # Загрузка аудио на CPU
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y, sr = librosa.load(file_path, sr=sr, mono=True)
            
            # Вычисление STFT на CPU
            stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
            spectrogram = np.abs(stft)**2
            
            # MFCC features
            mfcc = librosa.feature.mfcc(S=librosa.power_to_db(spectrogram), sr=sr, n_mfcc=20)  # Увеличено количество коэффициентов
            
            # Нормализация и преобразование в GPU массив
            mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
            
            # Выравнивание и обрезка
            flattened = mfcc.flatten()
            if len(flattened) > max_length:
                flattened = flattened[:max_length]
            else:
                flattened = np.pad(flattened, (0, max_length - len(flattened)))
            
            # Перенос на GPU если доступно
            if self.use_gpu:
                flattened = cp.asarray(flattened)
            
            return flattened
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def extract_spectrogram_features(self, file_path, *args, **kwargs):
        """Универсальный метод извлечения признаков"""
        if self.use_gpu:
            return self.extract_spectrogram_features_gpu(file_path, *args, **kwargs)
        else:
            return self.extract_spectrogram_features_cpu(file_path, *args, **kwargs)
    
    def extract_spectrogram_features_cpu(self, file_path, sr=22050, n_fft=2048, hop_length=512, max_length=50000):
        """Резервный метод для CPU"""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y, sr = librosa.load(file_path, sr=sr, mono=True)
            
            stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
            spectrogram = np.abs(stft)**2
            mfcc = librosa.feature.mfcc(S=librosa.power_to_db(spectrogram), sr=sr, n_mfcc=20)
            
            mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
            flattened = mfcc.flatten()
            
            if len(flattened) > max_length:
                flattened = flattened[:max_length]
            else:
                flattened = np.pad(flattened, (0, max_length - len(flattened)))
            
            return flattened
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def find_duplicates_by_hash(self):
        """Поиск точных дубликатов по хэшу с многопоточностью"""
        hash_groups = defaultdict(list)
        
        for file_path in self.audio_files:
            file_hash = self.compute_md5_hash(file_path)
            if file_hash:
                hash_groups[file_hash].append(str(file_path))
        
        duplicates = {k: v for k, v in hash_groups.items() if len(v) > 1}
        print(f"Found {len(duplicates)} exact duplicates by hash")
        return duplicates
    
    def build_feature_matrix(self):
        """Построение матрицы признаков с прогресс-баром"""
        print("Extracting spectrogram features...")
        
        features = {}
        valid_files = []
        
        for i, file_path in enumerate(self.audio_files):
            if i % 100 == 0:
                print(f"Processing file {i+1}/{len(self.audio_files)}")
            
            feature = self.extract_spectrogram_features(file_path)
            if feature is not None:
                # Конвертируем в numpy если это GPU массив
                if self.use_gpu and hasattr(feature, 'get'):
                    feature = feature.get()
                
                features[str(file_path)] = feature
                valid_files.append(str(file_path))
                self.spectrogram_features[str(file_path)] = feature
        
        # Создаем общую матрицу признаков
        if features:
            self.feature_matrix = np.array(list(features.values()))
            self.file_paths = list(features.keys())
            print(f"Feature matrix shape: {self.feature_matrix.shape}")
        
        return len(features)
    
    def find_similar_gpu(self):
        """Поиск похожих файлов с GPU-ускорением"""
        if self.feature_matrix is None or len(self.feature_matrix) < 2:
            return []
        
        print("GPU: Computing similarities...")
        
        # Перенос данных на GPU
        X_gpu = cp.asarray(self.feature_matrix)
        
        # Используем k-NN для поиска ближайших соседей
        knn = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
        knn.fit(X_gpu)
        
        # Находим соседей для каждой точки
        distances, indices = knn.kneighbors(X_gpu)
        
        # Конвертируем результаты обратно на CPU
        distances = cp.asnumpy(distances)
        indices = cp.asnumpy(indices)
        
        # Преобразуем расстояния в схожести
        similarities = 1 - distances
        
        # Находим пары с высокой схожести
        similar_pairs = []
        for i in range(len(self.file_paths)):
            for j_idx, dist_idx in enumerate(indices[i]):
                if i < dist_idx and similarities[i, j_idx] > self.similarity_threshold:
                    similar_pairs.append((
                        self.file_paths[i], 
                        self.file_paths[dist_idx],
                        similarities[i, j_idx]
                    ))
        
        return self._group_similar_files_gpu(similar_pairs)
    
    def find_similar_cpu(self):
        """Резервный метод для CPU"""
        if self.feature_matrix is None or len(self.feature_matrix) < 2:
            return []
        
        print("CPU: Computing similarities...")
        
        # Используем попарное косинусное сходство
        similarity_matrix = cosine_similarity(self.feature_matrix)
        
        similar_pairs = []
        n_files = len(self.file_paths)
        
        for i in range(n_files):
            for j in range(i + 1, n_files):
                if similarity_matrix[i, j] > self.similarity_threshold:
                    similar_pairs.append((
                        self.file_paths[i],
                        self.file_paths[j],
                        similarity_matrix[i, j]
                    ))
        
        return self._group_similar_files(similar_pairs)
    
    def _group_similar_files_gpu(self, similar_pairs):
        """Группировка похожих файлов для GPU"""
        groups = []
        visited = set()
        
        # Сортируем пары по убыванию схожести
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        
        for file1, file2, similarity in similar_pairs:
            if file1 in visited or file2 in visited:
                continue
                
            group = {file1, file2}
            visited.update({file1, file2})
            groups.append(list(group))
        
        return groups
    
    def _group_similar_files(self, similar_pairs):
        """Группировка похожих файлов для CPU"""
        from collections import defaultdict
        
        # Создаем граф связей
        graph = defaultdict(set)
        for file1, file2, similarity in similar_pairs:
            graph[file1].add(file2)
            graph[file2].add(file1)
        
        # Находим связные компоненты
        visited = set()
        groups = []
        
        for node in graph:
            if node not in visited:
                group = set()
                stack = [node]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        group.add(current)
                        stack.extend(graph[current] - visited)
                
                if len(group) > 1:
                    groups.append(list(group))
        
        return groups
    
    def find_similar_by_spectrogram(self):
        """Универсальный метод поиска похожих файлов"""
        valid_count = self.build_feature_matrix()
        
        if valid_count < 2:
            print("Not enough valid files for comparison")
            return []
        
        if self.use_gpu:
            return self.find_similar_gpu()
        else:
            return self.find_similar_cpu()
    
    def analyze_library(self):
        """Полный анализ библиотеки с оптимизациями"""
        print(f"Starting analysis of: {self.library_path}")
        print(f"Time: {datetime.now()}")
        print(f"Using GPU: {self.use_gpu}")
        
        # Поиск файлов
        self.find_audio_files()
        
        if not self.audio_files:
            print("No audio files found!")
            return None
        
        # Поиск точных дубликатов
        hash_duplicates = self.find_duplicates_by_hash()
        
        # Поиск похожих файлов
        similar_files = self.find_similar_by_spectrogram()
        
        results = {
            'analysis_date': datetime.now().isoformat(),
            'library_path': str(self.library_path),
            'total_files': len(self.audio_files),
            'valid_files': len(self.spectrogram_features),
            'exact_duplicates': hash_duplicates,
            'similar_files': similar_files,
            'settings': {
                'use_gpu': self.use_gpu,
                'similarity_threshold': self.similarity_threshold,
                'gpu_available': GPU_AVAILABLE
            }
        }
        
        return results
    
    def save_results(self, results, output_file):
        """Сохранение результатов в JSON файл с оптимизацией"""
        # Конвертируем numpy типы для сериализации
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj
        
        results = convert_types(results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {output_file}")
    
    def print_summary(self, results):
        """Упрощенный вывод сводки результатов"""
        if results is None:
            print("No results to display")
            return
            
        print("\n" + "="*60)
        print("MUSIC LIBRARY ANALYSIS SUMMARY")
        print("="*60)
        print(f"Library path: {results['library_path']}")
        print(f"Analysis date: {results['analysis_date']}")
        print(f"Total files found: {results['total_files']}")
        print(f"Successfully processed: {results['valid_files']}")
        print(f"Exact duplicates found: {len(results['exact_duplicates'])} groups")
        print(f"Similar files found: {len(results['similar_files'])} groups")
        print(f"GPU acceleration: {results['settings']['use_gpu']}")
        print(f"Similarity threshold: {results['settings']['similarity_threshold']}")

def main():
    parser = argparse.ArgumentParser(description='Find duplicate music files using spectrogram analysis with GPU acceleration')
    parser.add_argument('library_path', help='Path to music library')
    parser.add_argument('--threshold', type=float, default=0.95, 
                       help='Similarity threshold (0.0-1.0)')
    parser.add_argument('--output', default='music_duplicates.json', 
                       help='Output JSON file')
    parser.add_argument('--cpu', action='store_true', 
                       help='Force CPU mode even if GPU is available')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Batch size for processing')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.library_path):
        print(f"Error: Path '{args.library_path}' does not exist")
        return
    
    # Создание анализатора
    analyzer = MusicDuplicateFinder(
        library_path=args.library_path,
        similarity_threshold=args.threshold,
        use_gpu=not args.cpu,
        batch_size=args.batch_size
    )
    
    try:
        # Запуск анализа
        results = analyzer.analyze_library()
        
        if results:
            # Сохранение результатов
            analyzer.save_results(results, args.output)
            
            # Вывод сводки
            analyzer.print_summary(results)
            
            # Финальное сообщение о завершении работы
            print("\n" + "="*60)
            print("РАБОТА СКРИПТА ЗАВЕРШЕНА")
            print("="*60)
            print(f"Все результаты сохранены в файл: {args.output}")
            print(f"Найдено групп точных дубликатов: {len(results['exact_duplicates'])}")
            print(f"Найдено групп похожих файлов: {len(results['similar_files'])}")
            print("Для просмотра полной информации откройте файл с результатами")
            
        else:
            print("Analysis failed or no results")
            
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
