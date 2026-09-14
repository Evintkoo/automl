//! Text feature extraction

use crate::error::{AutoMLError, Result};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Simple text tokenizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextTokenizer {
    lowercase: bool,
    min_token_length: usize,
    stop_words: Vec<String>,
}

impl TextTokenizer {
    pub fn new() -> Self {
        Self {
            lowercase: true,
            min_token_length: 2,
            stop_words: Vec::new(),
        }
    }

    pub fn with_lowercase(mut self, lowercase: bool) -> Self {
        self.lowercase = lowercase;
        self
    }

    pub fn with_min_length(mut self, len: usize) -> Self {
        self.min_token_length = len;
        self
    }

    pub fn with_english_stop_words(mut self) -> Self {
        self.stop_words = vec![
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "is", "was", "are", "were", "be", "have", "has",
            "it", "this", "that", "i", "you", "he", "she", "we", "they",
        ].iter().map(|s| s.to_string()).collect();
        self
    }

    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let processed = if self.lowercase {
            text.to_lowercase()
        } else {
            text.to_string()
        };

        processed
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .filter(|s| s.len() >= self.min_token_length)
            .filter(|s| !self.stop_words.contains(&s.to_string()))
            .map(|s| s.to_string())
            .collect()
    }
}

impl Default for TextTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Count-based text vectorizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CountVectorizer {
    tokenizer: TextTokenizer,
    vocabulary: HashMap<String, usize>,
    max_features: Option<usize>,
    min_df: usize,
    max_df: f64,
    ngram_range: (usize, usize),
    binary: bool,
}

impl CountVectorizer {
    pub fn new() -> Self {
        Self {
            tokenizer: TextTokenizer::new().with_english_stop_words(),
            vocabulary: HashMap::new(),
            max_features: None,
            min_df: 1,
            max_df: 1.0,
            ngram_range: (1, 1),
            binary: false,
        }
    }

    pub fn with_max_features(mut self, n: usize) -> Self {
        self.max_features = Some(n);
        self
    }

    pub fn with_ngram_range(mut self, min: usize, max: usize) -> Self {
        self.ngram_range = (min.max(1), max.max(min));
        self
    }

    fn generate_ngrams(&self, tokens: &[String]) -> Vec<String> {
        let mut ngrams = Vec::new();
        
        for n in self.ngram_range.0..=self.ngram_range.1 {
            if tokens.len() >= n {
                for i in 0..=(tokens.len() - n) {
                    let ngram = tokens[i..i + n].join(" ");
                    ngrams.push(ngram);
                }
            }
        }
        
        ngrams
    }

    pub fn fit(&mut self, documents: &[String]) -> Result<()> {
        let n_docs = documents.len();
        let max_df_count = (self.max_df * n_docs as f64).ceil() as usize;

        let mut doc_freq: HashMap<String, usize> = HashMap::new();

        for doc in documents {
            let tokens = self.tokenizer.tokenize(doc);
            let ngrams = self.generate_ngrams(&tokens);
            
            let unique: std::collections::HashSet<&String> = ngrams.iter().collect();
            for ngram in unique {
                *doc_freq.entry(ngram.clone()).or_insert(0) += 1;
            }
        }

        let mut filtered: Vec<(String, usize)> = doc_freq
            .into_iter()
            .filter(|(_, count)| *count >= self.min_df && *count <= max_df_count)
            .collect();

        filtered.sort_by(|a, b| b.1.cmp(&a.1));

        if let Some(max_n) = self.max_features {
            filtered.truncate(max_n);
        }

        self.vocabulary.clear();
        for (idx, (term, _)) in filtered.into_iter().enumerate() {
            self.vocabulary.insert(term, idx);
        }

        Ok(())
    }

    pub fn transform(&self, documents: &[String]) -> Result<Array2<f64>> {
        if self.vocabulary.is_empty() {
            return Err(AutoMLError::ValidationError(
                "Vectorizer not fitted".to_string()
            ));
        }

        let n_docs = documents.len();
        let n_features = self.vocabulary.len();
        let mut result = Array2::zeros((n_docs, n_features));

        for (doc_idx, doc) in documents.iter().enumerate() {
            let tokens = self.tokenizer.tokenize(doc);
            let ngrams = self.generate_ngrams(&tokens);

            let mut counts: HashMap<&str, f64> = HashMap::new();
            for ngram in &ngrams {
                *counts.entry(ngram.as_str()).or_insert(0.0) += 1.0;
            }

            for (term, &idx) in &self.vocabulary {
                if let Some(&count) = counts.get(term.as_str()) {
                    result[[doc_idx, idx]] = if self.binary { 1.0 } else { count };
                }
            }
        }

        Ok(result)
    }

    pub fn fit_transform(&mut self, documents: &[String]) -> Result<Array2<f64>> {
        self.fit(documents)?;
        self.transform(documents)
    }

    pub fn get_feature_names(&self) -> Vec<String> {
        let mut names = vec![String::new(); self.vocabulary.len()];
        for (term, &idx) in &self.vocabulary {
            names[idx] = term.clone();
        }
        names
    }
}

impl Default for CountVectorizer {
    fn default() -> Self {
        Self::new()
    }
}

/// TF-IDF vectorizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TfidfVectorizer {
    count_vectorizer: CountVectorizer,
    idf: Option<Array1<f64>>,
    normalize: bool,
    smooth_idf: bool,
    sublinear_tf: bool,
}

impl TfidfVectorizer {
    pub fn new() -> Self {
        Self {
            count_vectorizer: CountVectorizer::new(),
            idf: None,
            normalize: true,
            smooth_idf: true,
            sublinear_tf: false,
        }
    }

    pub fn with_max_features(mut self, n: usize) -> Self {
        self.count_vectorizer = self.count_vectorizer.with_max_features(n);
        self
    }

    pub fn with_ngram_range(mut self, min: usize, max: usize) -> Self {
        self.count_vectorizer = self.count_vectorizer.with_ngram_range(min, max);
        self
    }

    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    pub fn fit(&mut self, documents: &[String]) -> Result<()> {
        self.count_vectorizer.fit(documents)?;

        let count_matrix = self.count_vectorizer.transform(documents)?;
        let n_docs = documents.len() as f64;
        let n_features = count_matrix.ncols();

        let mut idf = Array1::zeros(n_features);

        for j in 0..n_features {
            let df = count_matrix.column(j)
                .iter()
                .filter(|&&v| v > 0.0)
                .count() as f64;

            idf[j] = if self.smooth_idf {
                ((n_docs + 1.0) / (df + 1.0)).ln() + 1.0
            } else {
                (n_docs / df.max(1.0)).ln() + 1.0
            };
        }

        self.idf = Some(idf);
        Ok(())
    }

    pub fn transform(&self, documents: &[String]) -> Result<Array2<f64>> {
        let idf = self.idf.as_ref().ok_or_else(|| {
            AutoMLError::ValidationError("Vectorizer not fitted".to_string())
        })?;

        let mut tf_matrix = self.count_vectorizer.transform(documents)?;

        if self.sublinear_tf {
            tf_matrix.mapv_inplace(|v| if v > 0.0 { 1.0 + v.ln() } else { 0.0 });
        }

        let n_docs = tf_matrix.nrows();
        for i in 0..n_docs {
            for j in 0..tf_matrix.ncols() {
                tf_matrix[[i, j]] *= idf[j];
            }
        }

        if self.normalize {
            for i in 0..n_docs {
                let norm: f64 = tf_matrix.row(i).iter().map(|&v| v * v).sum::<f64>().sqrt();
                if norm > 0.0 {
                    for j in 0..tf_matrix.ncols() {
                        tf_matrix[[i, j]] /= norm;
                    }
                }
            }
        }

        Ok(tf_matrix)
    }

    pub fn fit_transform(&mut self, documents: &[String]) -> Result<Array2<f64>> {
        self.fit(documents)?;
        self.transform(documents)
    }

    pub fn get_feature_names(&self) -> Vec<String> {
        self.count_vectorizer.get_feature_names()
    }
}

impl Default for TfidfVectorizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Hashing vectorizer (memory efficient)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HashingVectorizer {
    n_features: usize,
    tokenizer: TextTokenizer,
    ngram_range: (usize, usize),
    normalize: bool,
}

impl HashingVectorizer {
    pub fn new(n_features: usize) -> Self {
        Self {
            n_features: n_features.max(1),
            tokenizer: TextTokenizer::new().with_english_stop_words(),
            ngram_range: (1, 1),
            normalize: true,
        }
    }

    pub fn with_ngram_range(mut self, min: usize, max: usize) -> Self {
        self.ngram_range = (min.max(1), max.max(min));
        self
    }

    fn hash_term(&self, term: &str) -> usize {
        let mut hash: u64 = 5381;
        for byte in term.bytes() {
            hash = ((hash << 5).wrapping_add(hash)).wrapping_add(byte as u64);
        }
        (hash as usize) % self.n_features
    }

    fn generate_ngrams(&self, tokens: &[String]) -> Vec<String> {
        let mut ngrams = Vec::new();
        
        for n in self.ngram_range.0..=self.ngram_range.1 {
            if tokens.len() >= n {
                for i in 0..=(tokens.len() - n) {
                    let ngram = tokens[i..i + n].join(" ");
                    ngrams.push(ngram);
                }
            }
        }
        
        ngrams
    }

    pub fn transform(&self, documents: &[String]) -> Array2<f64> {
        let n_docs = documents.len();
        let mut result = Array2::zeros((n_docs, self.n_features));

        for (doc_idx, doc) in documents.iter().enumerate() {
            let tokens = self.tokenizer.tokenize(doc);
            let ngrams = self.generate_ngrams(&tokens);

            for ngram in ngrams {
                let idx = self.hash_term(&ngram);
                result[[doc_idx, idx]] += 1.0;
            }

            if self.normalize {
                let norm: f64 = result.row(doc_idx).iter().map(|&v| v * v).sum::<f64>().sqrt();
                if norm > 0.0 {
                    for j in 0..self.n_features {
                        result[[doc_idx, j]] /= norm;
                    }
                }
            }
        }

        result
    }
}

impl Default for HashingVectorizer {
    fn default() -> Self {
        Self::new(1024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer() {
        let tokenizer = TextTokenizer::new()
            .with_lowercase(true)
            .with_min_length(2);

        let tokens = tokenizer.tokenize("Hello World! TEST.");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"test".to_string()));
    }

    #[test]
    fn test_count_vectorizer() {
        let docs = vec![
            "hello world hello".to_string(),
            "world test world".to_string(),
        ];

        let mut vectorizer = CountVectorizer::new()
            .with_max_features(10);

        let result = vectorizer.fit_transform(&docs).unwrap();
        assert_eq!(result.nrows(), 2);
    }

    #[test]
    fn test_tfidf_vectorizer() {
        let docs = vec![
            "machine learning great".to_string(),
            "deep learning powerful".to_string(),
        ];

        let mut vectorizer = TfidfVectorizer::new()
            .with_max_features(20)
            .with_normalize(true);

        let result = vectorizer.fit_transform(&docs).unwrap();
        assert_eq!(result.nrows(), 2);
    }
}
