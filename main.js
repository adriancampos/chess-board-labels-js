const class_names = [
    "blank",
    "pawn (Black)",
    "knight (Black)",
    "bishop (Black)",
    "rook (Black)",
    "queen (Black)",
    "king (Black)",
    "pawn (White)",
    "knight (White)",
    "bishop (White)",
    "rook (White)",
    "queen (White)",
    "king (White)"
]
const label_indices = [
    "0",
    "p",
    "n",
    "b",
    "r",
    "q",
    "k",
    "P",
    "N",
    "B",
    "R",
    "Q",
    "K"
]


let numCellsH = 8;
let numCellsV = 8;

function loadImage() {
    var reader = new FileReader();

    reader.onload = function (e) {
        document.getElementById('imgContainer').setAttribute('src', e.target.result);
    };

    reader.readAsDataURL(document.getElementById('imginput').files[0])

    clearPredictionLabels();
}

function displayFakePredictions() {
    let indices = [[0, 3, 0, 3, 0, 0, 9, 6], [0, 0, 0, 0, 0, 2, 0, 0], [12, 0, 0, 0, 0, 0, 0, 0], [0, 8, 0, 9, 0, 0, 0, 0], [0, 0, 10, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]];
    displayPredictions(indices);
}

function displayPredictions(indices) {


    clearPredictionLabels();


    console.log(indices);


    for (let row = 0; row < indices.length; row++) {
        rowElement = document.createElement("div");
        rowElement.className = 'row';

        predictionContainer.appendChild(rowElement);

        for (let cell = 0; cell < indices[row].length; cell++) {
            const prediction = indices[row][cell];

            // Convert piece index to label (use '' for blank piece)
            let predictionLabel = '';
            if (prediction != 0) {
                // predictionLabel = label_indices[prediction];
                predictionLabel = class_names[prediction];
            }
            cellElement = document.createElement("div");
            cellElement.className = 'cell';
            cellElement.textContent = predictionLabel;
            rowElement.appendChild(cellElement);
        }

    }
}

function clearPredictionLabels() {
    predictionContainer = document.getElementById('predictionContainer');

    while (predictionContainer.firstChild) {
        predictionContainer.removeChild(predictionContainer.firstChild);
    }
}

function doPrediction() {
    let indices = predict(document.getElementById('imgContainer'));

    displayPredictions(indices);    
}

function test1() {
    // c = 0;
    // for (let i = 0; i < 1000000000; i++) {
    //     c += 1;
    // }

    // console.log(c);
    predict(document.getElementById('imgContainer'));



    // function work() {
    //     c = 0;
    //     for (let i = 0; i < 100000000; i++) {
    //         c += 1;
    //     }

    //     console.log(c);
    // }

    // let b = new Blob(["onmessage =" + work.toString()], { type: "text/javascript" });
    // let worker = new Worker(URL.createObjectURL(b));
    // worker.postMessage(null);
    // // return await new Promise(resolve => worker.onmessage = e => resolve(e.data));


    // return new Promise(resolve => {
    //         c = 0;
    // for (let i = 0; i < 1000000000; i++) {
    //     c+=1;
    // }
    // });



    // Build a worker from an anonymous function body



}

function predict(imgData) {
    return tf.tidy(() => {
        const pred = model.predict(preprocess(imgData),{batchSize: 1}).arraySync(); // TODO batchSize is being weird. Any high number makes the predictions all close to 0 for certain cells
        console.log(pred);

        const indices = tf.argMax(pred, 1).reshape([8, 8]).arraySync();
        console.log(indices);

        return indices;        
    });
}

function preprocess(img) {
    // Convert image to tensor
    let tensor = tf.browser.fromPixels(img)

    // Resize so that we can nicely walk over the cells
    tensor = tf.image.resizeBilinear(tensor, [400, 400]).toFloat()

    // Turn the 64 squares into individual images
    let cellHeight = tensor.shape[0] / numCellsV;
    let cellWidth = tensor.shape[1] / numCellsH;

    // TODO: I think there's a numpy way to do this nicely. Similar to spaceToBatchND, but without the interlacing
    let slices = [];
    for (let i = 0; i < numCellsV; i++) {
        for (let j = 0; j < numCellsH; j++) {
            slices.push(
                tensor.slice([cellHeight * i, cellWidth * j], [cellHeight, cellWidth])
            );
        }
    }

    // Make a 4d tensor out of those 64 images
    let cells = tf.stack(slices);
    console.log(cells.shape)

    // Resize each image to 227 by 227
    // TODO: It'd be nice for our model to not require these huge images
    cells = tf.image.resizeBilinear(cells, [227, 227]).toFloat()

    // Normalize images
    cells = cells.div(tf.scalar(255.0));
    console.log(cells.shape)

    return cells
}



document.addEventListener("DOMContentLoaded", async function () {
    // Load model
    model = await tf.loadLayersModel('models/alexnet/model.json')
    document.getElementById('loadStatus').remove();
    document.getElementById('predict').removeAttribute('disabled');
});