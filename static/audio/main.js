window.AudioContext = window.AudioContext || window.webkitAudioContext;

var audioContext = new AudioContext();
var audioInput = null,
    realAudioInput = null,
    inputPoint = null,
    recording = false;
var rafID = null;
var analyserContext = null;
var canvasWidth, canvasHeight;

const wakeWords = ["hey", "fourth", "brain", "oov"]
const bufferSize = 1024
const channels = 1
const windowSize = 750
const zmuv_mean = 0.000016
const zmuv_std = 0.072771
const bias = 1e-7
const batches = 1
const SPEC_HOP_LENGTH = 200;
const MEL_SPEC_BINS = 40;
const NUM_FFTS = 512;
const audioFloatSize = 32767

let predictWords = []
let arrayBuffer = []
let targetState = 0

const windowBufferSize = Math.ceil(SAMPLE_RATE / bufferSize * windowSize /1000) * 1000

let session;
async function loadModel() {
    session = new onnx.InferenceSession();
    await session.loadModel("static/audio/onnx_model.onnx");
}
loadModel()

const addprediction = function(word) {
    words = document.createElement('p');
    words.innerHTML = '<b>' + word + '</b>';
    document.getElementById('wavefiles').appendChild(words);
}

function toggleRecording( e ) {
    if (e.classList.contains('recording')) {
        // stop recording
        e.classList.remove('recording');
        recording = false;
    } else {
        // start recording
        e.classList.add('recording');
        recording = true;
    }
}

function convertToMono( input ) {
    var splitter = audioContext.createChannelSplitter(2);
    var merger = audioContext.createChannelMerger(2);

    input.connect( splitter );
    splitter.connect( merger, 0, 0 );
    splitter.connect( merger, 0, 1 );
    return merger;
}

function cancelAnalyserUpdates() {
    window.cancelAnimationFrame( rafID );
    rafID = null;
}

function updateAnalysers(time) {
    if (!analyserContext) {
        var canvas = document.getElementById('analyser');
        canvasWidth = canvas.width;
        canvasHeight = canvas.height;
        analyserContext = canvas.getContext('2d');
    }

    // analyzer draw code here
    {
        var SPACING = 3;
        var BAR_WIDTH = 1;
        var numBars = Math.round(canvasWidth / SPACING);
        var freqByteData = new Uint8Array(analyserNode.frequencyBinCount);

        analyserNode.getByteFrequencyData(freqByteData); 

        analyserContext.clearRect(0, 0, canvasWidth, canvasHeight);
        analyserContext.fillStyle = '#F6D565';
        analyserContext.lineCap = 'round';
        var multiplier = analyserNode.frequencyBinCount / numBars;

        // Draw rectangle for each frequency bin.
        for (var i = 0; i < numBars; ++i) {
            var magnitude = 0;
            var offset = Math.floor( i * multiplier );
            // gotta sum/average the block, or we miss narrow-bandwidth spikes
            for (var j = 0; j< multiplier; j++)
                magnitude += freqByteData[offset + j];
            magnitude = magnitude / multiplier;
            var magnitude2 = freqByteData[i * multiplier];
            analyserContext.fillStyle = "hsl( " + Math.round((i*360)/numBars) + ", 100%, 50%)";
            analyserContext.fillRect(i * SPACING, canvasHeight, BAR_WIDTH, -magnitude);
        }
    }
    
    rafID = window.requestAnimationFrame( updateAnalysers );
}

function flatten(log_mels) {
    flatten_arry = []
    for(i = 0; i < log_mels.length; i++) {
        for(j = 0; j < log_mels[i].length; j++) {
            flatten_arry.push((Math.log(log_mels[i][j] + bias) - zmuv_mean) / zmuv_std)
        }
    }
    return flatten_arry
}

function argMax(array) {
  return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}

function softmax(arr) {
    return arr.map(function(value,index) { 
      return Math.exp(value) / arr.map( function(y /*value*/){ return Math.exp(y) } ).reduce( function(a,b){ return a+b })
    })
}

function gotStream(stream) {
    inputPoint = audioContext.createGain();

    // Create an AudioNode from the stream.
    realAudioInput = audioContext.createMediaStreamSource(stream);
    audioInput = realAudioInput;

    audioInput = convertToMono( audioInput );
    audioInput.connect(inputPoint);

    analyserNode = audioContext.createAnalyser();
    analyserNode.fftSize = 2048;
    inputPoint.connect( analyserNode );

    // bufferSize, in_channels, out_channels
    scriptNode = (audioContext.createScriptProcessor || audioContext.createJavaScriptNode).call(audioContext, bufferSize, channels, channels);
    scriptNode.onaudioprocess = async function (audioEvent) {
        if (recording) {
            let resampledMonoAudio = await resampleAndMakeMono(audioEvent.inputBuffer);
            arrayBuffer = [...arrayBuffer, ...resampledMonoAudio]

            // if we got 750 ms seconds of buffer
            if (arrayBuffer.length >= windowBufferSize) {
                // trim if it is more than 750 ms
                if (arrayBuffer.length / SAMPLE_RATE * 1000 > windowSize) {
                    arrayBuffer = arrayBuffer.slice(0, windowBufferSize)
                }
                // arrayBuffer = arrayBuffer.filter(x => x/audioFloatSize)
                // calculate log mels
                log_mels = melSpectrogram(arrayBuffer, {
                    sampleRate: SAMPLE_RATE,
                    hopLength: SPEC_HOP_LENGTH,
                    nMels: MEL_SPEC_BINS,
                    nFft: NUM_FFTS,
                    fMin: 60,
                });
                // convert to nd array
                let nd_mels = ndarray(flatten(log_mels), [MEL_SPEC_BINS, log_mels.length])
                // create empty [1,1,40,61]
                let dataProcessed = ndarray(new Float32Array(MEL_SPEC_BINS * log_mels.length * channels), [1, channels, MEL_SPEC_BINS, log_mels.length])
                // fill last 2 dims - 40 x 61 - [1, 1, 40, 61]
                ndarray.ops.assign(dataProcessed.pick(0, 0, null, null), nd_mels.pick(null,  null));
                let inputTensor = new onnx.Tensor(dataProcessed.data, 'float32', dataProcessed.shape);
                // Run model with Tensor inputs and get the result.
                let outputMap = await session.run([inputTensor]);
                let outputData = outputMap.values().next().value.data;
                let scores = Array.from(outputData)
                let probs = softmax(scores)
                probs = probs.filter(x => x/probs.reduce( (sum, x) => x+sum))
                let class_idx = argMax(probs)
                console.log(wakeWords[class_idx])
                console.log(probs)
                if (wakeWords[targetState] == wakeWords[class_idx]) {
                    console.log(wakeWords[class_idx])
                    addprediction(wakeWords[class_idx])
                    predictWords.push(word) 
                    targetState += 1
                    if (wakeWords.join(' ') == predictWords.join(' ')) {
                        addprediction(predictWords.join(' '))
                        predictWords = []
                        targetState = 0
                    }
                }
                arrayBuffer = []
            }
        }
    }
    inputPoint.connect(scriptNode);
    scriptNode.connect(audioContext.destination);

    zeroGain = audioContext.createGain();
    zeroGain.gain.value = 0.0;
    inputPoint.connect( zeroGain );
    zeroGain.connect( audioContext.destination );
    updateAnalysers();
}

function initAudio() {
    if (!navigator.getUserMedia)
        navigator.getUserMedia = navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
    if (!navigator.cancelAnimationFrame)
        navigator.cancelAnimationFrame = navigator.webkitCancelAnimationFrame || navigator.mozCancelAnimationFrame;
    if (!navigator.requestAnimationFrame)
        navigator.requestAnimationFrame = navigator.webkitRequestAnimationFrame || navigator.mozRequestAnimationFrame;

    navigator.getUserMedia({audio: true}, gotStream, function(e) {
        alert('Error getting audio');
        console.log(e);
    });
}

window.addEventListener('load', initAudio );