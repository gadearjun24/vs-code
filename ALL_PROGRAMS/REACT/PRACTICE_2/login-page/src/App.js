import "./App.css";
import "./components/SignUp.css";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import SignUp from "./components/SignUp";
import Login from "./components/Login";
import Dashboard from "./components/Dashboard";
import Home from "./components/Home";
import axios from "axios";
import { useState } from "react";

function App() {
  const [verify, setVerify] = useState(false);
  async function fetchData() {
    const token = localStorage.getItem("token");
    // console.log(token);

    const res = await axios.post(
      "https://ominous-disco-r4g77g74rjrqhwqq6-8080.app.github.dev/users/verify",
      { token: token },
      {
        header: {
          "Content-Type": "Application/json",
        },
      }
    );

    setVerify(res.data);
  }
  fetchData();

  return (
    <>
      <Router>
        <Routes>
          <Route path="/" element={<Home />}></Route>
          <Route path="/signup" element={<SignUp />}></Route>
          <Route path="/login" element={<Login />}></Route>
          <Route
            path="/dashboard"
            element={<Dashboard verify={verify} fetchData={fetchData} />}
          ></Route>
        </Routes>
      </Router>
    </>
  );
}

export default App;
