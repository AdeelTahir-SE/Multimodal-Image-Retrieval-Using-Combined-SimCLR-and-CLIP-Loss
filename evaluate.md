METRICS TO EVALUATE:
Recall@K (MOST IMPORTANT)

Measures: Is the correct match in the top K results?

R@1 → % of queries where correct item is top result
R@5 → % where correct item is in top 5
R@10 → % where correct item is in top 10

👉 Use for:

Image → Image
Text → Image

Mean Reciprocal Rank (MRR)

Measures: How early does the correct result appear?

MRR=
N
1
	​

∑
rank
i
	​

1
	​

If correct result is rank 1 → score = 1
Rank 10 → score = 0.1

👉 Better than Recall because it rewards earlier hits



Mean Average Precision (mAP)

Measures: Ranking quality when multiple correct matches exist

Useful if:

Multiple captions per image
Similar images count as correct



Precision@K

Measures: How many of top K results are actually relevant

Less used than Recall@K in retrieval papers, but still useful.


    Note:RUN this on data/coco/val2017 (you can get captions form annotations)

    When you are running evaluation give me query and result then i will tell form results how many it gave correct results for both text and images ok?

    Alos use faiss for fast


    Architecture:
    use model for embedding images after that store them in faiss
    then give tests like a query and image then run on that and give relevant images (ui with tellingmatching images number )i will tell matching after all those tests you calculate metrics ok?

    model(checkpoint):checkpoints/best.pt
