import VI
import numpy as np
from sklearn import svm

"""
    Experiment description:
    
    In the text classiﬁcation problem, we wish to classify a document into two or more mutually exclusive classes. 
    As in any classiﬁcation problem, we may wish to consider generative approaches or discriminative approaches. 
    In particular, by using one LDA module for each class, we obtain a generative model for classiﬁcation. 
    It is also of interest to use LDA in the discriminative framework, and this is our focus in this section. 
    
    A challenging aspect of the document classiﬁcation problem is the choice of features. 
    Treating individual words as features yields a rich but very large feature set (Joachims, 1999). 
    One way to reduce this feature set is to use an LDA model for dimensionality reduction. In particular, 
    LDA reduces any document to a ﬁxed set of real-valued features—the posterior Dirichlet parameters γ∗(w) 
    associated with the document. It is of interest to see how much discriminatory information we lose in 
    reducing the document description to these parameters. 
    
    ***We want to classify a document into two or more mutually exclusive classes.
    ***Choice of features. Using many words for features gives rich, but large feature set. Use LDA model for 
    dimensionality.
    ***How much discriminatory info do we lose??
    
    Experiment instructions:
    
    We conducted two binary classiﬁcation experiments 
    using the Reuters-21578 dataset. The dataset contains 8000 documents and 15,818 words. 

    In these experiments, we estimated the parameters of an LDA model on all the documents, without reference to 
    their true class label. We then trained a support vector machine (SVM) on the low-dimensional representations 
    provided by LDA and compared this SVM to an SVM trained on all the word features. 
    
    Using the SVMLight software package (Joachims, 1999), 
    we compared an SVM trained on all the word features with those trained on features induced by a 50-topic LDA model. 
    Note that we reduce the feature space by 99.6 percent in this case.

"""