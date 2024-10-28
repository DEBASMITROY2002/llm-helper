from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
from typing import List


class TextProcessor:
    def __init__(self, lowe_case=False, 
                 lemmatization=False, 
                 stem=False, 
                 stop_word_removal=False, 
                 split_pattern=r'[ ,.]', 
                 white_space_replace_pattern=r'\s+'):
        
        self.lowe_case = lowe_case
        self.lemmatization = lemmatization
        self.stem = stem
        self.stop_word_removal = stop_word_removal

        self.split_pattern = split_pattern
        self.white_space_replace_pattern = white_space_replace_pattern

        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def remove_white_space(self, text: str) -> str:
        return re.sub(self.white_space_replace_pattern, ' ', text)
    
    def lemmatize(self, words: List[str]) -> List[str]:
        return [self.lemmatizer.lemmatize(word) for word in words]

    def stemm(self, words: List[str]) -> List[str]:
        return [self.stemmer.stem(word) for word in words]

    def remove_stop_words(self, words: List[str]) -> List[str]:
        return [word for word in words if word not in self.stop_words]

    def process_text(self, text: str) -> List[str]:
        if self.lowe_case:
            text = text.lower()
        words = re.split(self.split_pattern, text)
        if self.lemmatization:
            words = self.lemmatize(words)
        if self.stem:
            words = self.stemm(words)
        if self.stop_word_removal:
            words = self.remove_stop_words(words)
        
        processed_text = ' '.join(words)
        processed_text =  self.remove_white_space(processed_text)
        processed_text =  self.remove_white_space(processed_text) # remove white space again
        return processed_text
    

if __name__ == '__main__':
    src_file = './dict_v0.txt'
    tar_file = './dict_v0_processed.txt'

    processor = TextProcessor(lowe_case=True,
                                lemmatization=True,
                                stem=True,
                                stop_word_removal=False)
    
    with open(src_file, 'r') as f:
        lines = f.readlines()
        processed_lines = [processor.process_text(line.replace('\n', '')) for line in lines]
        lines = list(set(processed_lines)) + lines

    print('Writing to file...', len(lines))

    with open(tar_file, 'w') as f:
        f.write('\n'.join(lines))

    print('Done!') 

