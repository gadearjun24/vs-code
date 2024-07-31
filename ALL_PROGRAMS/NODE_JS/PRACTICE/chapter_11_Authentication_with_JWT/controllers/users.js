const jwt = require("jsonwebtoken");
const userModel = require("../models/users");
const User = userModel.User;

// create new user
exports.createNewUser = async (req, res) => {
  try {
    const newUser = await new User(req.body);
    const token = jwt.sign({ email: req.body.email }, "sk");
    newUser.token = token;
    await newUser.save();
    res.json(newUser);
  } catch (err) {
    res.json({ err });
  }
};

// get all users
exports.getAllUsers = async (req, res) => {
  try {
    const users = await User.find({});
    res.json(users);
  } catch (err) {
    res.json({ err });
  }
};

// get user by id
exports.getOneUser = async (req, res) => {
  try {
    const id = req.params.id;
    const user = await User.findOne({ _id: id });
    res.json(user);
  } catch (err) {
    res.json({ err });
  }
};

// update user by id
exports.updateUser = async (req, res) => {
  try {
    const id = req.params.id;
    const user = await User.findByIdAndUpdate({ _id: id }, req.body, {
      new: true,
    });
    res.json(user);
  } catch (err) {
    res.json({ err });
  }
};

// delete user by id
exports.deleteUser = async (req, res) => {
  try {
    const id = req.params.id;
    const user = await User.findByIdAndDelete({ _id: id });
    res.json(user);
  } catch (err) {
    res.json({ err });
  }
};
