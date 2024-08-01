import { useNavigate } from "react-router-dom";
import "./App.css";
import "./components/SignUp.css";
function App() {
  const navigate = useNavigate();

  function handleClick(e) {
    navigate("/login");
  }

  return (
    <>
      <center>
        <h1>WELCOME</h1>
        <br />
        <br />
        <button onClick={handleClick}>Login</button>
      </center>
    </>
  );
}

export default App;
