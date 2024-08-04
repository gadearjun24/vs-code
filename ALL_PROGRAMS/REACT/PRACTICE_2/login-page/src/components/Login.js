import axios from "axios";
import { useNavigate } from "react-router-dom";
import "./SignUp.css";
import { useState } from "react";

function Login() {
  async function handelLoginCheck() {
    try {
      var res = await axios.post(
        "https://ominous-disco-r4g77g74rjrqhwqq6-8080.app.github.dev/users/login",
        info,
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      if (res.status === 200) {
        console.log(res.data);
        localStorage.setItem("token", res.data[1].token);
        navigate("/dashboard");
      }
    } catch (err) {
      console.log({ err });
    }
  }

  const navigate = useNavigate();
  const [info, setInfo] = useState({});

  function handelOnChange(e) {
    setInfo({ ...info, [e.target.name]: e.target.value });
  }

  function handleSignUpClick(e) {
    navigate("/signup");
  }

  return (
    <>
      <div className="container">
        <label>Email : </label>
        <input
          onChange={handelOnChange}
          name="email"
          type="email"
          placeholder="Enter email"
        ></input>
        <br />
        <label>Password : </label>
        <input
          onChange={handelOnChange}
          name="password"
          type="password"
          placeholder="Enter the password"
        ></input>
        <div className="btn">
          <button onClick={handelLoginCheck}>Login</button>
          <label onClick={handleSignUpClick}>you haven't account</label>
        </div>
      </div>
    </>
  );
}
export default Login;
