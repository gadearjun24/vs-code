const express = require("express");
const userController = require("../controllers/users");

const router = express.Router();

router.post("/signup/", userController.signUp).post('/login/' , userController.login);
// .get("/", userController.getAllUsers)
// .get("/:id", userController.getOneUser)
// .patch("/:id", userController.updateUser)
// .delete("/:id", userController.deleteUser);

exports.router = router;
