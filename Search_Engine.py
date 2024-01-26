from json import load
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from num2words import num2words
from nltk.stem import PorterStemmer
import math
import functools,collections,operator 
import difflib
import string

stopwords = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()



class Utilities:
  
  @staticmethod
  def RemovePunctuation(text):
   # Create a translation table to replace punctuation with spaces
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    
    # Use translate to replace punctuation with spaces
    text_with_spaces = text.translate(translator)
    
    return text_with_spaces
  
  @staticmethod
  def RemoveApostrophe(text:str):
    return np.char.replace(text, "'", "")

  @staticmethod
  def RemoveSingleCharacters(text:str):
    words = Utilities.GetWordsListOfText(text)
    new_text = ""
    for w in words:
      if len(w) > 1:
         new_text = new_text + " " + w
    return new_text

  @staticmethod
  def ConvertNumberstoLetters(text:str):
    words = Utilities.GetWordsListOfText(text)
    new_text = ''
    for w in words:
      if str.isnumeric(w):
        new_text+=' '+num2words(w)
      else:
        new_text+=' '+w
    return new_text
  
  @staticmethod
  def RemoveStopWords(text:str):
    words = Utilities.GetWordsListOfText(text)
    new_text = ""
    for word in words:
      if word not in stopwords:
         new_text += " " + word
    return new_text
  
  @staticmethod
  def Stemmer(text:str):
    words = Utilities.GetWordsListOfText(text)
    new_text = ""
    for word in words:
         new_text += " " + ps.stem(lemmatizer.lemmatize(word))
    return new_text

  @staticmethod
  def GetWordsListOfText(text:str):
    return nltk.word_tokenize(text)
  
  @staticmethod
  def Get_Paragraphs(text:str):
    pars = text.split('\n')
    return [Paragraph(pars[i],i+1) for i in range(len(pars))]

  @staticmethod
  def ProccessText(text:str):
    result = text.lower()
    result= Utilities.RemovePunctuation(result)
    # result= Utilities.RemoveApostrophe(result)
    # result = Utilities.RemoveSingleCharacters(result)
    # result = Utilities.ConvertNumberstoLetters(result)
    # result = Utilities.RemoveStopWords(result)
    # result = Utilities.Stemmer(result)
    # result= Utilities.RemovePunctuation(result)
    # result = Utilities.ConvertNumberstoLetters(result)
    return result
  
  @staticmethod
  def ReadDocuments(doc_ids):
    docs = []
    for item in doc_ids:
      try:
        with open('data/document_'+str(item)+'.txt') as f:
          data = f.read()
        docs.append(Document(data,item))
      except:
        ...
    return docs
  
  @staticmethod
  def CalculateCousineSimilarity(vector1,vector2):
    '''
    this function gets two vectores and calculates cousine similarity between them
    '''

    SigmaAiBi = sum({key : vector1[key]*vector2[key] for key in vector1}.values()) 
    SigmaAi2 = sum({key:vector1[key]*vector1[key] for key in vector1}.values())
    SigmaBi2 = sum({key:vector2[key]*vector2[key] for key in vector2}.values())
    return SigmaAiBi/(math.sqrt(SigmaAi2)*math.sqrt(SigmaBi2))


class Paragraph:
  def __init__(self,raw_data,par_num=None)-> None:
    self.RawData = raw_data
    self.Par_Num = par_num
    self.Proccessed_data = Utilities.ProccessText(raw_data)
    self.Proccessed_Words = Utilities.GetWordsListOfText(self.Proccessed_data)
    self.TF = None
    self.IDF=  None
    self.TF_IDF=  None



class Document:
  def __init__(self,raw_data,doc_id=None) -> None:
    self.RawData = raw_data
    self.Doc_ID = doc_id
    self.Paragraphs = Utilities.Get_Paragraphs(raw_data)
    self.Proccessed_Words = []
    for par in self.Paragraphs:
      for  word in par.Proccessed_Words:
        self.Proccessed_Words.append(word)

    self.TF = None
    self.IDF=  None
    self.TF_IDF=  None
    self.RepeativeWords = None
    self.ImportantWords = None




  
# notice that you should use Proccessed Term to Calculate its TF_IDF Because we have Proccessed Word lst for Both Document and Paragraph
# a singleton class that Performs TF_IDF Calculations : 
class TF_IDF:
  
  TFIDF = None
  
  def __new__(cls):
    cls.TFIDF=super().__new__(cls) if cls.TFIDF==None else cls.TFIDF
    return cls.TFIDF

  def __init__(self) -> None:
    pass

  def Calculate_TF(self,term:str,par:Paragraph,doc:Document):
    # TF of a term in paragraph =  number of Term in Paragraph/number of Terms in Whole Doc
    return par.Proccessed_Words.count(term)/len(doc.Proccessed_Words)


  def Calculate_DF(self,term,doc_lst):
    # gets number of Documents that term Exists
    return len(list(filter(lambda doc:doc.Proccessed_Words.count(term)>0,doc_lst)))
  

  def Calculate_IDF(self,term,doc_lst):
   df =  self.Calculate_DF(term,doc_lst)
   return math.log10(len(doc_lst)/(df+1))
  







class SearchEngine:
  
  Engine = None
  def __new__(cls,*args,**kwargs):

    cls.Engine =  super().__new__(cls) if cls.Engine==None else cls.Engine
    return cls.Engine

  def __init__(self,query,docs) -> None:
    # a singleton class thar performs Main Operations of our SearchEngine
    self.QueryDoc = Document(query)
    self.Doduments = docs
    self.Vocab = set()
    self.GetVocabSet()
    self.IDF = dict.fromkeys(self.Vocab,0)
    self.Calculate_IDF_for_VocabWords()
    self.Calculate_TFIDF_for_DocumentsList(docs)
    

  
  def Calculate_TFIDF_for_DocumentsList(self,docs):
    '''
   you will give list of your document to this function and 
   it will Calculate TF and TF_IDF for every document and its Paragraphs using two functions Calculate_TFIDF_for_Paragraph and 
   Calculate_TFIDF_for_Document
    '''
    for doc in docs:
      for par in doc.Paragraphs:
        self.Calculate_TFIDF_for_Paragraph(par,doc)
      self.Calculate_TFIDF_for_Document(doc)
    
      

  def Calculate_TFIDF_for_Paragraph(self,par:Paragraph,doc:Document):
    tfidf= TF_IDF()
    TFIDF_dict =  dict.fromkeys(self.Vocab,0)
    
    for w in par.Proccessed_Words:
      if TFIDF_dict.get(w)!=None:
         TFIDF_dict[w] =  tfidf.Calculate_TF(w,par,doc)

      else:
       
       closestMaches = difflib.get_close_matches(w,self.Vocab)
       tfidf_value = tfidf.Calculate_TF(w,par,doc)
       for item in closestMaches:
         
          TFIDF_dict[item] =  tfidf_value
    
    par.TF = dict.copy(TFIDF_dict)
    

    for w in TFIDF_dict:
      TFIDF_dict[w]*=self.IDF[w]

    par.TF_IDF = TFIDF_dict

  
  def Calculate_TFIDF_for_Document(self,doc:Document):

    tf_dict = dict.fromkeys(self.Vocab,0)
    tfidf_dict = dict.fromkeys(self.Vocab,0)

    
    for key in self.Vocab:
      for par in doc.Paragraphs:
        tf_dict[key]+=par.TF[key]
        tfidf_dict[key]+=par.TF_IDF[key]
    doc.TF= tf_dict
    doc.TF_IDF = tfidf_dict



  def Calculate_IDF_for_VocabWords(self):
    tfidf = TF_IDF()
    for w in self.IDF:
      self.IDF[w] = tfidf.Calculate_IDF(w,self.Doduments)
    
      
    
  def GetVocabSet(self):
    for doc in self.Doduments:
      for w in doc.Proccessed_Words:
        self.Vocab.add(w)

  def Search(self):
     for par in self.QueryDoc.Paragraphs:
       self.Calculate_TFIDF_for_Paragraph(par,self.QueryDoc)
     self.Calculate_TFIDF_for_Document(self.QueryDoc)

    #  Fetch the most Repeative and Important words in Doc_list
     for doc in self.Doduments:
       
       self.FetchTheMostRepeativeWordsOfDoc(doc)
       self.FetchTheMostImportantWordsOfDoc(doc)
     
        # Up to this line we have tf_idf vectores of docs and query and after this we should search for similarity between these vectores

     SimilarityDicts=self.CalculateCousineSimilarityBetweenDocsandQuery()
     SimilarityDicts= list(sorted(SimilarityDicts.items(),key=lambda item:-item[1]))
     for item in SimilarityDicts[0:10]:
       print(f'Doc_ID : {item[0].Doc_ID}\n')
       print(f"Repeative Words : {str.join(' - ',item[0].RepeativeWords[0:10])}")
       print(f"Important Words : {str.join(' - ',item[0].ImportantWords[0:10])}\n")
     
       ParSimilarities =  self.CalculateCousineSimilarityBetweenParagraphsandQuery(item[0])
       ParSimilarities = list(sorted(ParSimilarities.items(),key=lambda item:-item[1]))
       print(f'Paragraph {ParSimilarities[0][0].Par_Num} :  \n\n {ParSimilarities[0][0].RawData}\n')
       print(f'Paragraph {ParSimilarities[1][0].Par_Num} :  \n\n {ParSimilarities[1][0].RawData}\n')
       print(f'Paragraph {ParSimilarities[2][0].Par_Num} :  \n\n {ParSimilarities[2][0].RawData}')

       print(100*'-')
     
      
 
     
     
  
  

  
  
  def CalculateCousineSimilarityBetweenDocsandQuery(self):
    
    return {doc:Utilities.CalculateCousineSimilarity(self.QueryDoc.TF_IDF,doc.TF_IDF) for doc in self.Doduments}
  
  def CalculateCousineSimilarityBetweenParagraphsandQuery(self,doc:Document):

    return {par:Utilities.CalculateCousineSimilarity(self.QueryDoc.TF_IDF,par.TF_IDF) for par in doc.Paragraphs}

  
  def FetchTheMostRepeativeWordsOfDoc(self,doc:Document):
    sortedTF = sorted(doc.TF.items(),key=lambda item:-item[1])
    doc.RepeativeWords=  [item[0] for item in sortedTF]
    
  def FetchTheMostImportantWordsOfDoc(self,doc:Document):
    sortedTF_IDF = sorted(doc.TF_IDF.items(),key=lambda item:-item[1])
    doc.ImportantWords =  [item[0] for item in sortedTF_IDF]
    
  
    


# Driver_Code : 

if __name__=="__main__":
   
#    Load Data from Data.json
   with open('data.json') as f:
      data:list = load(f)
   
#    Get Query form User and Fetch Query Info From data list
      
      # Important Tip
# Notice that it can has more than one Occurance for a query with different informations and it is a bug in data.json and we Always Get idx 0 of query_infor lst
  
   query=  input()
   query_info_lst = [item for item in data if item['query']==query]
   query_info = query_info_lst[0] if len(query_info_lst)>0 else None
   
   if query_info is not None :

    Candidate_Docs = query_info['candidate_documents_id']
    DocumentsList=Utilities.ReadDocuments(Candidate_Docs)  
    Engine = SearchEngine(query_info['query'],DocumentsList)


    
    Engine.Search()
       
   else:
     print("N/A")
   


