from projects.ml_inference.inference import predict 

def test_predict_setosa():
    sample =[5.1,3.5,1.4,0.2]

    result = predict(sample)

    assert result =="setosa"

    
