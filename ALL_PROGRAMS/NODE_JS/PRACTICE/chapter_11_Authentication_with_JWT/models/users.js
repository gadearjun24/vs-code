const { Schema, mongoose } = require("mongoose");

const userSchema = new Schema({
  "first name": { type: String, required: true },
  "last name": { type: String },
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true, minimum: 6 },
  token: { type: String },
});

const User = mongoose.model("User", userSchema);

exports.User = User;
