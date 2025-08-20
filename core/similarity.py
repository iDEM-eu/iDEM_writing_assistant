from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
import torch,numpy
# Install BLEURT via pip install git+https://github.com/lucadiliello/bleurt-pytorch.git


BATCH_SIZE = 16

class SimilarityBLEURT(object):
    def __init__(self,device):
        pretrained_model = 'lucadiliello/BLEURT-20'
        self.bleurt_config = BleurtConfig.from_pretrained(pretrained_model)
        self.bleurt_model = BleurtForSequenceClassification.from_pretrained(pretrained_model).to(device)
        self.bleurt_tokenizer = BleurtTokenizer.from_pretrained(pretrained_model)
        self.device = device
    
    def assess(self, pairs):
        references = [pair[0] for pair in pairs]
        modified = [pair[1] for pair in pairs]
        references_batched = [references[i:i + BATCH_SIZE] for i in range(0, len(references), BATCH_SIZE)]
        modified_batched = [modified[i:i + BATCH_SIZE] for i in range(0, len(modified), BATCH_SIZE)]
        result = []
        for i in range(len(references_batched)):
            self.bleurt_model.eval()
            with torch.no_grad():
                inputs = self.bleurt_tokenizer(references_batched[i], modified_batched[i], padding='longest',
                                               max_length=512, truncation=True,
                                               return_tensors='pt')
                inputs = {key: inputs[key].to(self.device) for key in inputs}
                res = self.bleurt_model(**inputs).logits.flatten().to(torch.device('cpu')).tolist()
                print(res)
                result.extend(res)
        return numpy.array(result)
        