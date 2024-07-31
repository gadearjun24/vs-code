const express = require("express");
const jwt = require("jsonwebtoken");
const productController = require("../controllers/products");

const auth = (req, res, next) => {
  try {
    let token = req.headers["authorization"].split(" ")[1];
    console.log(token);
    let verify = jwt.verify(token, "sk");
    console.log(verify);
    if (verify) {
      next();
    } else {
      res.sendStatus(401);
    }
  } catch (err) {
    res.json({ err });
  }
};

const router = express.Router();

router
  .get("/", auth, productController.getAllProducts)
  .get("/:id", productController.getOneProduct)
  .post("/", productController.createProduct)
  .delete("/:id", productController.deleteProduct)
  .patch("/:id", productController.updateProduct);

exports.router = router;
