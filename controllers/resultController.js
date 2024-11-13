const Result = require('../models/Result'); // Import the Result model

const saveResult = async (req, res) => {
    const { email, result } = req.body;

    try {
        const updatedResult = await Result.findOneAndUpdate(
            { email },
            { $push: { results: result } },
            { new: true, upsert: true }
        );

        res.status(200).json({ message: 'Result saved successfully', data: updatedResult });
    } catch (error) {
        console.error('Error saving result:', error); // Log the error details
        res.status(500).json({ error: 'Failed to save result', details: error.message });
    }
};


module.exports = { saveResult };
