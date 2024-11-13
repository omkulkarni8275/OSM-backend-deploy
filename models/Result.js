const mongoose = require('mongoose');

const answerSchema = new mongoose.Schema({
    Age: { type: Number, required: true },
    'Duration of Symptoms (months)': { type: Number, required: true },
    'Obsession Type': { type: String, required: true },
    'Compulsion Type': { type: String, required: true },
    'Depression Diagnosis': { type: Number, required: true },
    'Anxiety Diagnosis': { type: Number, required: true },
});

const resultSchema = new mongoose.Schema({
    email: { type: String, required: true },
    results: [
        {
            severity: { type: String, required: true },  // if severity is a label like "Severe"
            percentage: { type: Number, required: true },
            answers: { type: answerSchema, required: true },
        },
    ],
});

module.exports = mongoose.model('Result', resultSchema);
