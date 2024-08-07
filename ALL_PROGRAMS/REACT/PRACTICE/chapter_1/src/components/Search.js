import "./Search.css";
import Suggetion from "./Suggetion";
import { useState } from "react";

function Search() {
  const [searchValue, setSearchValue] = useState([]);
  const [suggetion, setSuggetion] = useState([]);

  const values = ["a", "b", "c", "aa", "bb", "cc", "ab", "aabcde", "abcd"];

  function handleOnChange(e) {
    console.log(e);

    setSearchValue((searchValue) => [ e.target.value]);

    // console.log("search value", searchValue);

    setSuggetion(
      values.filter((ele, index) => {
        return ele.includes(searchValue);
      })
    );
  }

  return (
    <>
      <div className="container">
        <input
          name="search"
          type="text"
          placeholder="Search for suggetion"
          className="search"
          onChange={handleOnChange}
        ></input>
      </div>
      <Suggetion suggetion={suggetion}></Suggetion>
    </>
  );
}

export default Search;
