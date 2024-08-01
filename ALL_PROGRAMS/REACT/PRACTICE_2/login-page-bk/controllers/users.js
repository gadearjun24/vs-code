const userModel = require("../models/users");
const bcrypt = require("bcrypt");
const jwt = require("jsonwebtoken");

const User = userModel.User;

exports.get = async (req, res) => {
  res.json(await User.find({}));
};

exports.signUp = async (req, res) => {
  try {
    const newUser = await new User(req.body);
    const password = await bcrypt.hash(req.body.password, 10);
    newUser.password = password;
    const token = jwt.sign(req.body.password, "sk");
    newUser.token = token;
    await newUser.save();
    res.json(newUser);
  } catch (err) {
    res.json({ err });
  }
};
exports.login = async (req, res) => {
  try {
    const email = req.body.email;
    const password = req.body.password;
    const user = await User.findOne({ email: email });
    if (user && email && password) {
      const verify = await bcrypt.compare(password, user.password);
      console.log(email, password);

      console.log(verify);
      if (verify) {
        res.json(["user is authenticated", user]);
      } else {
        res.json("check for 'email' or password");
      }
    } else {
      res.json("user not found please sign up or check 'email'");
    }
  } catch (err) {
    res.json({ err });
  }
};
