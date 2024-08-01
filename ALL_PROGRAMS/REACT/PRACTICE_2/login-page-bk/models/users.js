const { mongoose, Schema } = require("mongoose");

const userSchema = new Schema({
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true, min: [6] },
  token: { type: String },
});

const User = mongoose.model("User", userSchema);

exports.User = User;
