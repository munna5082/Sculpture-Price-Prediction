from django.shortcuts import render
import pickle
import numpy as np

with open('sculpturepredmodel.pkl', 'rb')as file:
    model = pickle.load(file)

# Create your views here.
def home(request):
    try:
        if request.method == 'POST':
            name = request.POST['Name']
            reputation = request.POST['Repute']
            height = request.POST['Height']
            width = request.POST['Width']
            weight = request.POST['Weight']
            material = request.POST['Material']
            baseprice = request.POST['BasePrice']
            baseshippingprice = request.POST['BaseShippingPrice']
            expressshipment = request.POST['ExpressShipment']
            transport = request.POST['Transport']

            data = np.array([[float(reputation), float(height), float(width), float(weight), float(material), float(baseprice), float(baseshippingprice), float(expressshipment), float(transport)]])
            print(data)
            res = int(model.predict(data))

            return render(request, 'output.html', {'response' : res})
    except:
        return render(request, 'index.html', {'error' : 'Please fill all values'})

    return render(request, 'index.html')