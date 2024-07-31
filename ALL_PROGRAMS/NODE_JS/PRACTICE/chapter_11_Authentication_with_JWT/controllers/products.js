const productModel = require("../models/products");

const Product = productModel.Product;

// create new product
exports.createProduct = async (req, res) => {
  try {
    const newProduct = await new Product(req.body);
   await newProduct.save();
    res.json(newProduct);
  } catch (err) {
    res.json({ err });
  }
};

// get all products
exports.getAllProducts = async (req, res) => {
  try {
    const allProducts = await Product.find({});
    res.json(allProducts);
  } catch (err) {
    res.json({ err });
  }
};

// get one product by id
exports.getOneProduct = async (req, res) => {
  try {
    const id = req.params.id;
    const product = await Product.findOne({ _id: id });
    res.json(product);
  } catch (err) {
    [res.json({ err })];
  }
};

// delete product by id
exports.deleteProduct = async (req, res) => {
  try {
    const id = req.params.id;
    const product = await Product.findByIdAndDelete({ _id: id });
    res.json(product);
  } catch (error) {
    res.json({ error });
  }
};

// update product by id
exports.updateProduct = async (req, res) => {
  try {
    const id = req.params.id;
    const product = await Product.findByIdAndUpdate({ _id: id }, req.body , {new : true});
    res.json(product);
  } catch (error) {
    res.json({ error });
  }
};
