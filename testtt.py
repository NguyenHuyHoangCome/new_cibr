from my_tools.index import index_one
from my_tools.search import Search


features = index_one("D:\\HocTapEPU\\CBIR\\CBIRThi\\static\\images\\0eG0Lgx3dq2c.jpg")
print(features)



searcher = Search('my_tools/lfcnn/filecsv/lfcnn_ICA_n.csv')
results = searcher.search(features)

# results = searcher.gaborSearch(features)
RESULTS_LIST = list()
for (score, pathImage) in results:
    RESULTS_LIST.append(
        {"image": str(pathImage), "score": str(score)}
    )
print(RESULTS_LIST)
