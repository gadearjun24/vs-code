import "./Search.css";
import Suggetion from "./Suggetion";
import { useState } from "react";

function Search() {
  const [searchValue, setSearchValue] = useState([]);
  const [suggetion, setSuggetion] = useState([]);

  const values = ["a", "b", "c", "aa", "bb", "cc", "ab", "aabcde", "abcd"];

  function handleOnChange(e) {
    setSearchValue(
      (searchValue) => (searchValue = [...searchValue, e.target.value])
    );

    console.log("search value", searchValue);

    values.forEach((ele, index) => {
      console.log(
        ele,
        ele.charAt(searchValue.length - 1),
        ele.charAt(searchValue.length - 1) ===
          searchValue[searchValue.length - 1],
        searchValue[searchValue.length - 1],
        searchValue
      );
    //   setSuggetion(["a"]);

      if (
        ele.charAt(searchValue.length - 1) ===
        searchValue[searchValue.length - 1]
      ) {
        setSuggetion([...suggetion, ele]);
        console.log(suggetion);
      }
    });
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
