"""
The Cost of Ignoring Context in Sentiment Classification
========================================================
Empirical ablation of the bag-of-words assumption for sentiment classification
on formal (movie reviews) vs informal (tweets) text.

Model ladder:
  A. VADER lexicon (unsupervised baseline)
  B. TF-IDF unigrams      + Logistic Regression
  C. TF-IDF uni+bigrams   + Logistic Regression   (local context)
  D. C + negation marking + Logistic Regression   (linguistic feature)
  E. Word2Vec (trained)   + Logistic Regression   (distributional semantics)

Evaluation: 5-fold stratified cross-validation (accuracy + macro-F1).
Plus a fixed 75/25 split for confusion matrices and negation error analysis.

Run:  python src/pipeline.py
Outputs: results/results_cv.csv, results/results.json, figures/*.png
Everything is seeded (SEED=42) for exact reproduction.
"""
import json, re, random, warnings, time, os
import numpy as np, pandas as pd, nltk
from nltk.corpus import movie_reviews, twitter_samples
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from gensim.models import Word2Vec
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

warnings.filterwarnings("ignore"); random.seed(42); np.random.seed(42)
SEED = 42
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results"); FIG = os.path.join(HERE, "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)

for pkg in ["movie_reviews","twitter_samples","punkt","punkt_tab","stopwords","vader_lexicon"]:
    nltk.download(pkg, quiet=True)

tw_tok = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
NEG_WORDS = {"not","no","never","none","nobody","nothing","neither","nor","cannot","cant",
             "won't","wont","don't","dont","isn't","isnt","wasn't","wasnt","aren't","arent",
             "ain't","without","hardly","barely","scarcely","doesn't","doesnt","didn't",
             "didnt","couldn't","wouldn't","shouldn't"}
CLAUSE_END = {".",",",";",":","!","?","but"}
POS_EMO=[":)",":-)",":D",":-D","=)","=D",":]",":')","(:","(-:",":p",":P",";)",";-)","<3","^_^",":*",":>"]
NEG_EMO=[":(",":-(","=(","):",")-:",":[",":'(",";(",":/",":\\",":|",">:(",":-/",":-\\"]
ALL_EMO=sorted(set(POS_EMO+NEG_EMO), key=len, reverse=True)

def tokenize(t): return tw_tok.tokenize(t)
def negate_mark(toks):
    out, neg = [], False
    for t in toks:
        low=t.lower()
        if low in CLAUSE_END: neg=False; out.append(t); continue
        if low in NEG_WORDS or low.endswith("n't"): neg=True; out.append(t); continue
        out.append(t+"_NEG" if (neg and t.isalpha()) else t)
    return out
def clean_tweet(t):
    t=re.sub(r'https?://\S+|www\.\S+',' ',t); t=re.sub(r'@\w+',' ',t)
    for e in ALL_EMO: t=t.replace(e,' ')
    return re.sub(r'\s+',' ',t).strip()

class DS:
    """Pre-tokenised dataset so vectorisers never re-tokenise (fast + consistent)."""
    def __init__(self, pairs, cleaner=None):
        random.Random(SEED).shuffle(pairs)
        if cleaner: pairs=[(cleaner(t),y) for t,y in pairs]
        self.texts=[p[0] for p in pairs]; self.y=np.array([p[1] for p in pairs])
        self.toklists=[tokenize(t) for t in self.texts]
        self.plain=np.array([" ".join(tk) for tk in self.toklists])
        self.neg=np.array([" ".join(negate_mark(tk)) for tk in self.toklists])

def load_movie():
    return DS([(movie_reviews.raw(f),1) for f in movie_reviews.fileids('pos')]+
              [(movie_reviews.raw(f),0) for f in movie_reviews.fileids('neg')])
def load_tweets():
    p=twitter_samples.strings('positive_tweets.json'); n=twitter_samples.strings('negative_tweets.json')
    return DS([(t,1) for t in p]+[(t,0) for t in n], cleaner=clean_tweet)

_sia=SentimentIntensityAnalyzer()
def m_vader(ds,tr):
    return lambda te: np.array([1 if _sia.polarity_scores(ds.texts[i])["compound"]>=0 else 0 for i in te])
def _tfidf(field,ngram):
    def build(ds,tr):
        arr=getattr(ds,field)
        vec=TfidfVectorizer(token_pattern=r"\S+",ngram_range=ngram,min_df=2,sublinear_tf=True)
        clf=LogisticRegression(max_iter=2000,C=4.0,random_state=SEED)
        clf.fit(vec.fit_transform(arr[tr]),ds.y[tr])
        return lambda te: clf.predict(vec.transform(arr[te]))
    return build
def m_w2v(ds,tr):
    w2v=Word2Vec([ds.toklists[i] for i in tr],vector_size=100,window=5,min_count=2,workers=4,epochs=12,seed=SEED)
    dim=w2v.vector_size
    def dv(i):
        v=[w2v.wv[w] for w in ds.toklists[i] if w in w2v.wv]
        return np.mean(v,axis=0) if v else np.zeros(dim)
    clf=LogisticRegression(max_iter=2000,C=4.0,random_state=SEED)
    clf.fit(np.vstack([dv(i) for i in tr]),ds.y[tr])
    return lambda te: clf.predict(np.vstack([dv(i) for i in te]))

MODELS={"A. VADER lexicon":m_vader,"B. TF-IDF unigram":_tfidf("plain",(1,1)),
        "C. + bigrams":_tfidf("plain",(1,2)),"D. + negation":_tfidf("neg",(1,2)),
        "E. Word2Vec (trained)":m_w2v}

def evaluate(ds,name):
    skf=StratifiedKFold(5,shuffle=True,random_state=SEED); splits=list(skf.split(ds.plain,ds.y)); rows=[]
    for mn,b in MODELS.items():
        a,f=[],[]
        for tr,te in splits:
            p=b(ds,tr)(te); a.append(accuracy_score(ds.y[te],p)); f.append(f1_score(ds.y[te],p,average="macro"))
        rows.append(dict(dataset=name,model=mn,acc=float(np.mean(a)),acc_std=float(np.std(a)),
                         f1=float(np.mean(f)),f1_std=float(np.std(f))))
        print(f"  {mn:24s} acc={np.mean(a):.3f} f1={np.mean(f):.3f}",flush=True)
    return rows

def error_analysis(ds,name):
    idx=np.arange(len(ds.y)); tr,te=train_test_split(idx,test_size=0.25,stratify=ds.y,random_state=SEED)
    uni=_tfidf("plain",(1,1))(ds,tr)(te); neg=_tfidf("neg",(1,2))(ds,tr)(te); yte=ds.y[te]
    cms={"B. TF-IDF unigram":confusion_matrix(yte,uni).tolist(),"D. + negation":confusion_matrix(yte,neg).tolist()}
    fixed=[j for j in range(len(te)) if uni[j]!=yte[j] and neg[j]==yte[j]]
    cue=re.compile(r"\b(not|no|never|n't|without|hardly|barely)\b",re.I)
    wn=[j for j in fixed if cue.search(ds.texts[te[j]])]
    ex=[dict(text=ds.texts[te[j]][:150].replace("\n"," "),true=int(yte[j])) for j in wn[:6]]
    stats=dict(dataset=name,n_test=len(te),n_fixed=len(fixed),n_fixed_with_negation=len(wn),
               pct_fixed_with_negation=(len(wn)/len(fixed)*100 if fixed else 0))
    return cms,stats,ex

# ---- figures ---------------------------------------------------------------
NAVY="#1F3A5F"; TEAL="#238C8A"; GRAY="#8895A7"
def make_figures(R):
    cv=R["cv"]; DSN=["Movie Reviews (formal)","Tweets (informal)"]
    MS=["A. VADER lexicon","B. TF-IDF unigram","C. + bigrams","D. + negation","E. Word2Vec (trained)"]
    SH=["VADER\nlexicon","TF-IDF\nunigram","+ bigrams","+ negation","Word2Vec\n(trained)"]
    C={DSN[0]:NAVY,DSN[1]:TEAL}
    plt.rcParams.update({"font.size":13,"figure.dpi":220,"savefig.dpi":220})
    def v(d,m,k):
        for r in cv:
            if r["dataset"]==d and r["model"]==m: return r[k]
    def clean(ax):
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True,color="#E3E7EC",lw=1); ax.set_axisbelow(True)
    # accuracy
    fig,ax=plt.subplots(figsize=(9.2,5.0)); x=np.arange(len(MS)); w=0.38
    for i,d in enumerate(DSN):
        acc=[v(d,m,"acc")*100 for m in MS]; err=[v(d,m,"acc_std")*100 for m in MS]
        bars=ax.bar(x+(i-0.5)*w,acc,w,yerr=err,capsize=3,label=d,color=C[d],edgecolor="white",lw=0.6,error_kw=dict(ecolor="#333",lw=1.1))
        for b,a in zip(bars,acc): ax.text(b.get_x()+b.get_width()/2,a+1.2,f"{a:.1f}",ha="center",va="bottom",fontsize=10.5,fontweight="bold")
    ax.axhline(50,ls="--",lw=1,color=GRAY); clean(ax); ax.set_xticks(x); ax.set_xticklabels(SH,fontsize=11)
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(45,100)
    ax.set_title("Accuracy by model  ·  5-fold cross-validation",fontsize=15,fontweight="bold",loc="left",pad=12)
    ax.legend(frameon=False,loc="lower center",ncol=2,fontsize=12,bbox_to_anchor=(0.5,-0.28))
    plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig_accuracy.png"),bbox_inches="tight"); plt.close()
    # marginal
    fig,ax=plt.subplots(figsize=(9.2,5.0))
    steps=[("Supervised BoW\n(vs VADER)","B. TF-IDF unigram","A. VADER lexicon"),
           ("+ Bigrams\n(vs unigram)","C. + bigrams","B. TF-IDF unigram"),
           ("+ Negation\n(vs unigram)","D. + negation","B. TF-IDF unigram"),
           ("Word2Vec\n(vs unigram)","E. Word2Vec (trained)","B. TF-IDF unigram")]
    x=np.arange(len(steps)); w=0.38
    for i,d in enumerate(DSN):
        dd=[(v(d,b,"acc")-v(d,a,"acc"))*100 for _,b,a in steps]
        bars=ax.bar(x+(i-0.5)*w,dd,w,label=d,color=C[d],edgecolor="white",lw=0.6)
        for b,val in zip(bars,dd):
            ax.text(b.get_x()+b.get_width()/2,val+(0.6 if val>=0 else -0.6),f"{val:+.1f}",ha="center",va="bottom" if val>=0 else "top",fontsize=10.5,fontweight="bold")
    ax.axhline(0,color="#333",lw=1.2); clean(ax); ax.set_xticks(x); ax.set_xticklabels([s[0] for s in steps],fontsize=10.5)
    ax.set_ylabel("Change in accuracy (pp)"); ax.set_ylim(-16,32)
    ax.set_title("Where does the accuracy come from?",fontsize=15,fontweight="bold",loc="left",pad=12)
    ax.legend(frameon=False,loc="lower left",fontsize=11)
    plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig_marginal.png"),bbox_inches="tight"); plt.close()
    # confusion
    conf=R["confusion"]; fig,axes=plt.subplots(2,2,figsize=(7.6,7.4))
    pairs=[(DSN[0],"B. TF-IDF unigram"),(DSN[0],"D. + negation"),(DSN[1],"B. TF-IDF unigram"),(DSN[1],"D. + negation")]
    for ax,(d,m) in zip(axes.ravel(),pairs):
        cm=np.array(conf[d][m]); cmn=cm/cm.sum(1,keepdims=True); ax.imshow(cmn,cmap="Blues",vmin=0,vmax=1)
        for r in range(2):
            for c in range(2): ax.text(c,r,f"{cm[r,c]}\n{cmn[r,c]*100:.0f}%",ha="center",va="center",fontsize=12,fontweight="bold",color="white" if cmn[r,c]>0.5 else NAVY)
        ax.set_xticks([0,1]);ax.set_yticks([0,1]);ax.set_xticklabels(["neg","pos"]);ax.set_yticklabels(["neg","pos"])
        ax.set_xlabel("predicted");ax.set_ylabel("true");ax.set_title(f"{d.split(' (')[0]}\n{m}",fontsize=11,fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig_confusion.png"),bbox_inches="tight"); plt.close()

def main():
    t0=time.time(); rows=[];cms={};stats=[];exs={}
    for name,load in [("Movie Reviews (formal)",load_movie),("Tweets (informal)",load_tweets)]:
        print(f"\n=== {name} ===",flush=True); ds=load()
        rows+=evaluate(ds,name); c,s,e=error_analysis(ds,name); cms[name]=c;stats.append(s);exs[name]=e
        print(f"  error-analysis: {s['n_fixed']} fixed, {s['pct_fixed_with_negation']:.0f}% negation",flush=True)
    R={"cv":rows,"confusion":cms,"error_stats":stats,"examples":exs}
    pd.DataFrame(rows).to_csv(os.path.join(RES,"results_cv.csv"),index=False)
    json.dump(R,open(os.path.join(RES,"results.json"),"w"),indent=2)
    make_figures(R)
    print(f"\nDONE in {time.time()-t0:.0f}s -> results/ and figures/",flush=True)

if __name__=="__main__": main()
