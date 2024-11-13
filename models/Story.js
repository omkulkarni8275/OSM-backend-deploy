const mongoose = require('mongoose');

const storySchema = new mongoose.Schema({
    title: {
        type: String,
        required: true,
    },
    imageUrl: {
        type: String,  
        required: true,
    },
    chapters: [
        {
            chapterTitle: {
                type: String,
                required: true,
            },
            content: {
                type: String,
                required: true,
            },
            questions: [
                {
                    text: {
                        type: String,
                        required: true,
                    },
                    answerType: {
                        type: String,
                        default: "multiple-choice", // Defaults to multiple-choice
                    },
                    options: {
                        type: [String],  // Array for dynamic options
                        validate: {
                            validator: function(arr) {
                                return arr.length > 0; // Ensures at least one option is present
                            },
                            message: "Options array cannot be empty",
                        },
                        default: ["Yes", "No"],  // Default options, but can be customized
                    },
                },
            ],
        },
    ],
});

module.exports = mongoose.model('Story', storySchema);
