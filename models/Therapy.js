const mongoose = require('mongoose');

const chapterSchema = new mongoose.Schema({
  title: { type: String, required: true },
  exercise: { type: String, required: true },
  activity: { type: String, required: true }
});

const therapySchema = new mongoose.Schema({
  storyId: { type: mongoose.Schema.Types.ObjectId, ref: 'Story', required: true },
  title: { type: String, required: true },
  chapters: [chapterSchema]
});

module.exports = mongoose.models.Therapy || mongoose.model('Therapy', therapySchema);
