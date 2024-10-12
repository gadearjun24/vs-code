import "./Search.css";

function Suggetion({ suggetion, setSearchValue, newFunction }) {
  function handleOnClickSpan(e) {
    console.log(e.target.innerText);
    // searchValue(e.target.innerText);
    setSearchValue(e.target.innerText);
  }

  return (
    <>
      {suggetion === "" ? (
        ""
      ) : (
        <div className="suggetion">
          {suggetion.map((ele, index) => {
            if (index < 5) {
              return <span onClick={handleOnClickSpan}>{ele}</span>;
            } else {
              return "";
            }
          })}
        </div>
      )}
    </>
  );
}
export default Suggetion;
