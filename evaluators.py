from sklearn.metrics import f1_score

def compute_generator_accuracy(y_true, y_pred):
    y_true = y_true.cpu().numpy().flatten()
    y_pred = y_pred.cpu().numpy().flatten()
    return f1_score(y_true, y_pred, average='macro')

def generate_summary(model, input):
    summary = ''
    
    return summary
