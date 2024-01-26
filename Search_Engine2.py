from Search_Engine import *
import random



if __name__=="__main__":
    # read 2000 Random Docs
    doc_list = Utilities.ReadDocuments(random.sample(range(0,50000),100))
    engine = SearchEngine('',doc_list)
    engine.ClusterReducedVectorsAndSaveitinGraph(engine.ReduceTFIDFDimensionsforDocs())