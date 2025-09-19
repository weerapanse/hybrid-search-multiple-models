from fastembed import SparseTextEmbedding, TextEmbedding, LateInteractionTextEmbedding
import json
print('')
print('')
print('Dense Model ----------------------------------------------------')
for d in TextEmbedding.list_supported_models():
    print(d['model'])
print('')
print('')
print('Sparse Model ----------------------------------------------------')
for d in SparseTextEmbedding.list_supported_models():
    print(d['model'])
print('')
print('')
print('Rerank Model ----------------------------------------------------')
for d in LateInteractionTextEmbedding.list_supported_models():
    print(d['model'])
print('')
print('')