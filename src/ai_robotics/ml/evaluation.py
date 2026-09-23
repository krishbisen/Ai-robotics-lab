from sklearn.metrics import accuracy_score ,confusion_matrix, precision_score , recall_score ,f1_score

def evaluate_classification(model, X_test, y_test):
    predictions = model.predict(X_test)
    matrix = confusion_matrix(y_test, predictions)


    metrics ={
        "accuracy" : accuracy_score(y_test, predictions),
        "precision": precision_score(y_test,
                                     predictions,
                                     average ="weighted",
                                     zero_division =0,),
        "recall": recall_score(y_test,
                               predictions,
                               average ="weighted",
                               zero_division =0,),
        "f1": f1_score(y_test,
                        predictions,
                        average ="weighted",
                        zero_division =0,),
        "confusion_matrix":matrix,                

    }
    return metrics
